"""Domains the agent's Web Search may not reach, edited from the admin page and enforced by the
Web Search Gateway on two layers, each changed by API call with no redeploy:

1. the connector's admin exclude list on the target: results from a blocked domain, or its
   subdomains, never come back for any query, and a caller cannot override it. The list
   itself is the record of what is blocked;
2. a Cedar forbid in the gateway's AgentCore Policy engine: a search whose query or include
   filter names a blocked domain is denied before it runs (`-32002 Tool Execution Denied`).

The engine, its baseline permit and its mode (WEB_SEARCH_POLICY_MODE) belong to
scripts/websearch_policy.py. Who blocked a domain, and when, is kept beside it in the
identity table, for the page only.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import boto3

logger = logging.getLogger("router.websearch")

_GATEWAY = os.environ.get("WEB_SEARCH_GATEWAY_NAME", "")
_ENGINE = os.environ.get("WEB_SEARCH_POLICY_ENGINE_NAME", "")
_TARGET = "web-search-tool"
_FORBID = "forbid_blocked_domains"
_ACTION = 'AgentCore::Action::"web-search-tool___WebSearch"'
# A blocked domain is a literal in the Cedar statement: only a plain host name gets that far.
_DOMAIN = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+([a-z]{2,63}|xn--[a-z0-9-]{1,59})$")
MAX_DOMAINS = 100
_c = boto3.client("bedrock-agentcore-control", region_name=os.environ.get("WEB_SEARCH_REGION", "us-east-1")) \
    if _GATEWAY else None
_notes = boto3.resource("dynamodb").Table(os.environ["IDENTITY_TABLE_NAME"]) if _GATEWAY else None


class NotDeployed(Exception):
    """No Web Search gateway, or no policy engine on it: WEB_SEARCH=true and provision.sh first."""


class Rejected(Exception):
    """A change that cannot be made; the message says why."""


class Busy(Exception):
    """The previous change is still being applied; the same request succeeds shortly."""


def normalise(text: str) -> str | None:
    """A domain as typed (with or without scheme, path or a leading www.) → its host, or None."""
    d = str(text or "").strip().lower()
    d = re.sub(r"^[a-z]+://", "", d).split("/", 1)[0].split(":", 1)[0].rstrip(".")
    d = d[4:] if d.startswith("www.") else d       # subdomains are excluded with it anyway
    return d if _DOMAIN.match(d) else None


def forbid_statement(gateway_arn: str, domains: list[str]) -> str:
    """The forbid for `domains`, each matched by _DOMAIN, so input changes only which literals
    it compares with. A domain counts in the query only where a host name can start — the start,
    or after a space, `.`, `/` or `@` — so blocking x.com leaves netflix.com alone. `like` is
    case-sensitive, and a domain run into other text (关于x.com) is missed: this layer is a first
    line, the exclude list is what holds."""
    names = ", ".join(json.dumps(d) for d in domains)
    query = " ||\n  ".join(" || ".join(f"context.input.query like {json.dumps(p.format(d))}"
                                        for p in ("{}*", "* {}*", "*.{}*", "*/{}*", "*@{}*")) for d in domains)
    return (f'forbid(principal, action == {_ACTION}, resource == AgentCore::Gateway::"{gateway_arn}") when {{\n'
            f"  {query} ||\n"
            "  (context.input has filters && context.input.filters has domainFilter &&\n"
            "   context.input.filters.domainFilter has include &&\n"
            f"   context.input.filters.domainFilter.include.containsAny([{names}]))\n"
            "};")


def _pages(call, key, **kw):
    token = None
    while True:
        page = call(**kw, **({"nextToken": token} if token else {}))
        yield from page.get(key, [])
        token = page.get("nextToken")
        if not token:
            return


def _locate() -> tuple[dict, dict, str]:
    """→ (gateway, target, policy engine id), found by the names this deployment gives them."""
    if not _c:
        raise NotDeployed("Web Search is not configured")
    gw = next((g for g in _pages(_c.list_gateways, "items") if g["name"] == _GATEWAY), None)
    if not gw:
        raise NotDeployed("the Web Search gateway is not deployed")
    gid = gw["gatewayId"]
    tid = next((t["targetId"] for t in _pages(_c.list_gateway_targets, "items", gatewayIdentifier=gid)
                if t["name"] == _TARGET), None)
    pe = next((p["policyEngineId"] for p in _pages(_c.list_policy_engines, "policyEngines") if p["name"] == _ENGINE), None)
    if not tid or not pe:
        raise NotDeployed("the Web Search gateway has no target or policy engine yet")
    return _c.get_gateway(gatewayIdentifier=gid), _c.get_gateway_target(gatewayIdentifier=gid, targetId=tid), pe


def _excluded(target: dict) -> list[str]:
    values = target["targetConfiguration"]["mcp"]["connector"]["configurations"][0].get("parameterValues") or {}
    return list((values.get("domainFilter") or {}).get("exclude") or [])


def _valid(domains: list[str]) -> list[str]:
    # An entry set outside the page (console, CLI) never reaches the Cedar text unchecked.
    return sorted(d for d in domains if _DOMAIN.match(d))


def _note_key(domain: str) -> dict:
    return {"PK": f"WEBSEARCH#{_GATEWAY}", "SK": domain}


def _forbid(pe: str) -> dict | None:
    """The forbid policy with its definition and status, or None."""
    p = next((p for p in _pages(_c.list_policies, "policies", policyEngineId=pe) if p["name"] == _FORBID), None)
    if not p:
        return None
    try:
        return _c.get_policy(policyEngineId=pe, policyId=p["policyId"])
    except _c.exceptions.ResourceNotFoundException:
        return None


def _settled(pe: str, deadline: float) -> dict | None:
    """The forbid once no create, update or delete is in flight on it (each refuses the next with
    ConflictException, measured); a policy being deleted settles as None."""
    while True:
        p = _forbid(pe)
        status = (p or {}).get("status", "")
        if not status.endswith("ING"):
            return p
        if time.time() >= deadline:
            raise Busy("the previous change is still being applied")
        time.sleep(1)


def blocked() -> dict:
    """→ {domains: [{domain, by, at, valid}], mode, target, policy: {status, reason}}: what the
    gateway enforces now. Each layer is applied asynchronously; policy.status is APPLYING while the
    forbid catches up with a change, and OUT_OF_SYNC when it still does not match the list a
    minute later, which a sync() repairs."""
    gw, target, pe = _locate()
    domains = sorted(_excluded(target))
    notes = {}
    keys = [_note_key(d) for d in domains]
    for i in range(0, len(keys), 100):                 # BatchGetItem's limit
        got = _notes.meta.client.batch_get_item(RequestItems={_notes.name: {"Keys": keys[i:i + 100]}})
        notes.update({n["SK"]: n for n in got["Responses"].get(_notes.name, [])})
    valid = _valid(domains)
    p = _forbid(pe)
    status = (p or {}).get("status", "") or "NONE"
    # The first reason repeats the statement; the rest say what went wrong.
    reasons = (p or {}).get("statusReasons") or []
    if (status == "ACTIVE" and (not valid or p["definition"]["cedar"]["statement"] != forbid_statement(gw["gatewayArn"], valid))
            or status == "NONE" and valid):
        # The list is written first and the forbid follows within a request's 30 s: only a
        # mismatch that outlasts that is one.
        updated = target.get("updatedAt")
        recent = updated and (datetime.now(timezone.utc) - updated).total_seconds() < 60
        status = "APPLYING" if recent else "OUT_OF_SYNC"
    return {"domains": [{"domain": d, "by": notes.get(d, {}).get("by", ""), "at": notes.get(d, {}).get("at", ""),
                         "valid": d in valid} for d in domains],
            "mode": (gw.get("policyEngineConfiguration") or {}).get("mode", ""),
            "target": target.get("status", ""),
            "policy": {"status": status, "reason": "; ".join(reasons[1:] or reasons) if status.endswith("FAILED") else ""}}


def _delete_policy(pe: str, pid: str, deadline: float) -> None:
    # Deleting is asynchronous, and a policy of the same name cannot be created until it is gone.
    _c.delete_policy(policyEngineId=pe, policyId=pid)
    while time.time() < deadline:
        try:
            _c.get_policy(policyEngineId=pe, policyId=pid)
        except _c.exceptions.ResourceNotFoundException:
            return
        time.sleep(1)
    raise Busy("the previous rule is still being removed")


def _apply(domains: list[str]) -> None:
    """Make both layers enforce exactly `domains`: the exclude list first, since it is what holds,
    then the forbid, whose validation is left to finish on its own (blocked() reports it). Each is
    compared first and written only if it differs, so a call after a partial failure finishes."""
    gw, target, pe = _locate()
    deadline = time.time() + 22                        # inside API Gateway's 30 s
    if target.get("status") != "READY" or sorted(_excluded(target)) != sorted(domains):
        cfg = target["targetConfiguration"]
        cfg["mcp"]["connector"]["configurations"][0]["parameterValues"] = (
            {"domainFilter": {"exclude": domains}} if domains else {})
        _c.update_gateway_target(gatewayIdentifier=gw["gatewayId"], targetId=target["targetId"], name=target["name"],
                                 targetConfiguration=cfg,
                                 credentialProviderConfigurations=target["credentialProviderConfigurations"])
        # Applied asynchronously: until READY the target still reads, and filters, as before.
        while True:
            t = _c.get_gateway_target(gatewayIdentifier=gw["gatewayId"], targetId=target["targetId"])
            if t.get("status") == "READY" and sorted(_excluded(t)) == sorted(domains):
                break
            if t.get("status", "").endswith(("FAILED", "UNSUCCESSFUL")):
                raise RuntimeError(f"the web search target did not update: {t.get('status')}")
            if time.time() >= deadline - 12:
                raise Busy("the web search target is still updating")
            time.sleep(1)
    valid = _valid(domains)
    existing = _settled(pe, deadline)
    if not valid:
        if existing:
            _delete_policy(pe, existing["policyId"], deadline)
        return
    statement = forbid_statement(gw["gatewayArn"], valid)
    definition = {"cedar": {"statement": statement}}
    if existing and existing.get("status", "").endswith("FAILED"):
        _delete_policy(pe, existing["policyId"], deadline)
        existing = None
    if not existing:
        _c.create_policy(policyEngineId=pe, name=_FORBID, definition=definition, validationMode="FAIL_ON_ANY_FINDINGS")
    elif existing["definition"]["cedar"]["statement"] != statement:
        _c.update_policy(policyEngineId=pe, policyId=existing["policyId"], definition=definition,
                         validationMode="FAIL_ON_ANY_FINDINGS")


def block(domain: str, by: str) -> None:
    _, target, _ = _locate()
    current = _excluded(target)
    if domain not in current and len(current) >= MAX_DOMAINS:
        raise Rejected(f"at most {MAX_DOMAINS} domains can be blocked")
    if not _notes.get_item(Key=_note_key(domain)).get("Item"):
        _notes.put_item(Item={**_note_key(domain), "by": by, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    _apply(sorted({*current, domain}))
    logger.info("web search domain blocked: by=%s %s", by, domain)


def unblock(domain: str, by: str) -> bool:
    """`domain` is matched exactly as stored. Applied even when already gone from the list, so a
    second try finishes a removal whose forbid update failed."""
    _, target, _ = _locate()
    current = _excluded(target)
    _apply(sorted(set(current) - {domain}))
    _notes.delete_item(Key=_note_key(domain))
    logger.info("web search domain unblocked: by=%s %s", by, domain)
    return domain in current


def sync(by: str) -> None:
    """Re-apply the current list to both layers: what the page's retry does."""
    _, target, _ = _locate()
    _apply(sorted(_excluded(target)))
    logger.info("web search rules re-applied: by=%s", by)
