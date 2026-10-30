"""The Web Search Gateway's AgentCore Policy engine: created, given its baseline permit, and
attached in WEB_SEARCH_POLICY_MODE — enforce (default) or log_only, where decisions are
recorded and nothing is denied.

Which domains are blocked is not set here: admins edit them on the access page, and the router
(lambda/router/websearch.py) writes the target's exclude list and the engine's
`forbid_blocked_domains` policy by API, with no redeploy. This script leaves both alone.

Usage (from scripts/provision.sh): PREFIX=… GID=… REGION=us-east-1 python websearch_policy.py
"""
from __future__ import annotations

import os
import sys
import time

import boto3

# UpdateGateway replaces the whole configuration, so every field it holds is sent back.
_GATEWAY_FIELDS = ("name", "roleArn", "protocolType", "authorizerType", "protocolConfiguration",
                   "authorizerConfiguration", "description", "exceptionLevel", "kmsKeyArn")
# Written by the router; every other policy in the engine is this script's.
_ROUTER_OWNED = {"forbid_blocked_domains"}


def _wait(get, ready=("READY", "ACTIVE"), tries=60):
    for _ in range(tries):
        status = get().get("status")
        if status in ready:
            return
        if status and status.endswith("FAILED"):
            raise SystemExit(f"  failed: {status}")
        time.sleep(3)
    raise SystemExit("  timed out waiting for AgentCore")


def _pages(call, key, **kw):
    token = None
    while True:
        page = call(**kw, **({"nextToken": token} if token else {}))
        yield from page.get(key, [])
        token = page.get("nextToken")
        if not token:
            return


def main() -> None:
    e = os.environ
    mode = {"enforce": "ENFORCE", "log_only": "LOG_ONLY"}.get(e.get("WEB_SEARCH_POLICY_MODE", "enforce").lower())
    if not mode:
        raise SystemExit("  WEB_SEARCH_POLICY_MODE must be enforce or log_only")
    c = boto3.client("bedrock-agentcore-control", region_name=e["REGION"])
    gid = e["GID"]
    g = c.get_gateway(gatewayIdentifier=gid)
    keep = {k: g[k] for k in _GATEWAY_FIELDS if g.get(k) is not None}

    # The engine denies by default, so a baseline permit is needed; its "overly permissive"
    # finding is the point, hence ignored.
    name = f"{e['PREFIX'].replace('-', '_')}_websearch"
    engine = next((p for p in _pages(c.list_policy_engines, "policyEngines") if p["name"] == name), None)
    if not engine:
        engine = c.create_policy_engine(name=name, description="Domains the agent's Web Search may not reach")
    pe = engine["policyEngineId"]
    _wait(lambda: c.get_policy_engine(policyEngineId=pe))
    pe_arn = c.get_policy_engine(policyEngineId=pe)["policyEngineArn"]
    baseline = f'permit(principal, action, resource == AgentCore::Gateway::"{g["gatewayArn"]}");'
    existing = {p["name"]: p for p in _pages(c.list_policies, "policies", policyEngineId=pe)}
    # The engine is this deployment's own: a policy nobody here defines has no business there.
    for pname in set(existing) - {"baseline_permit"} - _ROUTER_OWNED:
        c.delete_policy(policyEngineId=pe, policyId=existing.pop(pname)["policyId"])
    p = existing.get("baseline_permit")
    if p and p.get("status") != "ACTIVE":
        c.delete_policy(policyEngineId=pe, policyId=p["policyId"])
        # Asynchronous: a policy of the same name cannot be created until this one is gone.
        for _ in range(40):
            try:
                c.get_policy(policyEngineId=pe, policyId=p["policyId"])
            except c.exceptions.ResourceNotFoundException:
                break
            time.sleep(1)
        p = None
    if not p or c.get_policy(policyEngineId=pe, policyId=p["policyId"])["definition"]["cedar"]["statement"] != baseline:
        definition = {"cedar": {"statement": baseline}}
        if p:
            c.update_policy(policyEngineId=pe, policyId=p["policyId"], definition=definition,
                            validationMode="IGNORE_ALL_FINDINGS")
            pid = p["policyId"]
        else:
            pid = c.create_policy(policyEngineId=pe, name="baseline_permit", definition=definition,
                                  validationMode="IGNORE_ALL_FINDINGS")["policyId"]
        _wait(lambda: c.get_policy(policyEngineId=pe, policyId=pid), ready=("ACTIVE",))
    if g.get("policyEngineConfiguration") != {"arn": pe_arn, "mode": mode}:
        c.update_gateway(gatewayIdentifier=gid, **keep, policyEngineConfiguration={"arn": pe_arn, "mode": mode})
        _wait(lambda: c.get_gateway(gatewayIdentifier=gid))
    print(f"  policy: {mode} on {pe}; blocked domains are set on the access page")


if __name__ == "__main__":
    sys.exit(main())
