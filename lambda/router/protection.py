"""Protection rules: which Lark documents and folders the agent may not touch.

Each rule is a template-linked policy in the Verified Permissions store (stacks/protection_stack.py):
a user's rule binds their own Lark account to a resource, an admin's binds nobody, so it holds
for everyone. Linking a template is all this module ever writes — no Cedar text is built from
input. The lark-cli MCP server reads the same store before every document call.

Whose rule it is comes from the caller's verified identity, never from the request: a user's
principal is their Lark account, and deleting a rule checks the stored principal first.

A template-linked policy carries no description, so what a person needs to recognise a rule —
the resource's title and link, and who added it — is kept beside it in the identity table,
keyed by policy id. The policy alone decides; the note is only for the page and /protected.
"""
from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse

import boto3

logger = logging.getLogger("router.protection")

_STORE = os.environ.get("POLICY_STORE_ID", "")
_ADMIN = os.environ.get("PROTECT_ADMIN_TEMPLATE_ID", "")
_USER = os.environ.get("PROTECT_USER_TEMPLATE_ID", "")
_avp = boto3.client("verifiedpermissions") if _STORE else None
_notes = boto3.resource("dynamodb").Table(os.environ["IDENTITY_TABLE_NAME"]) if _STORE else None

# The Lark hosts a document link lives on, tenant subdomains included.
_HOSTS = ("larksuite.com", "feishu.cn", "larkoffice.com")
_DOCUMENT_PATHS = {"docx", "docs", "sheets", "base", "file", "mindnotes", "slides"}
_TOKEN = re.compile(r"^[A-Za-z0-9]{8,64}$")


def configured() -> bool:
    return bool(_STORE and _ADMIN and _USER)


def parse_link(text: str) -> tuple[str, str] | None:
    """A Lark Drive link → ("document" | "folder", token), or None for anything else."""
    u = urlparse(str(text or "").strip())
    host = (u.hostname or "").lower()
    if u.scheme != "https" or not any(host == h or host.endswith("." + h) for h in _HOSTS):
        return None
    parts = [p for p in u.path.split("/") if p]
    if len(parts) >= 3 and parts[0] == "drive" and parts[1] == "folder" and _TOKEN.match(parts[2]):
        return "folder", parts[2]
    if len(parts) >= 2 and parts[0] in _DOCUMENT_PATHS and _TOKEN.match(parts[1]):
        return "document", parts[1]
    return None


def principal(app_id: str, open_id: str) -> dict:
    # Scoped to the app: an open_id is per app, and a rule must not follow a reused one elsewhere.
    return {"entityType": "Lark::User", "entityId": f"{app_id}/{open_id}"}


def _resource(kind: str, token: str) -> dict:
    return {"entityType": "Lark::Folder" if kind == "folder" else "Lark::Document", "entityId": token}


def _template(owner: dict | None) -> str:
    return _USER if owner else _ADMIN


def _note_key(policy_id: str) -> dict:
    return {"PK": f"PROTECTION#{_STORE}", "SK": policy_id}


def _attach_notes(found: list[dict]) -> list[dict]:
    keys = [_note_key(r["policyId"]) for r in found]
    notes = {}
    for i in range(0, len(keys), 100):                 # BatchGetItem's limit
        got = _notes.meta.client.batch_get_item(RequestItems={_notes.name: {"Keys": keys[i:i + 100]}})
        notes.update({n["SK"]: n for n in got["Responses"].get(_notes.name, [])})
    for r in found:
        n = notes.get(r["policyId"], {})
        r.update(title=n.get("title", ""), url=n.get("url", ""), by=n.get("by", ""))
    return found


def note(policy_id: str, kind: str, token: str, title: str, url: str, by: str) -> None:
    _notes.put_item(Item={**_note_key(policy_id), "kind": kind, "token": token,
                          "title": title, "url": url, "by": by})


def rules(owner: dict | None) -> list[dict]:
    """A user's rules (owner = their principal), or the admins' (owner = None), with their notes."""
    flt = {"policyTemplateId": _template(owner), **({"principal": {"identifier": owner}} if owner else {})}
    out, token = [], None
    while True:
        page = _avp.list_policies(policyStoreId=_STORE, filter=flt, **({"nextToken": token} if token else {}))
        for p in page.get("policies", []):
            r = p.get("resource") or {}
            out.append({"policyId": p["policyId"], "createdDate": str(p.get("createdDate", "")),
                        "kind": "folder" if r.get("entityType") == "Lark::Folder" else "document",
                        "token": r.get("entityId", "")})
        token = page.get("nextToken")
        if not token:
            return _attach_notes(out)


def protect(owner: dict | None, kind: str, token: str, by: str, title: str = "", url: str = "") -> str:
    """Link the rule, unless the same one exists; returns its policy id."""
    for r in rules(owner):
        if r["kind"] == kind and r["token"] == token:
            return r["policyId"]
    linked = {"policyTemplateId": _template(owner), "resource": _resource(kind, token)}
    if owner:
        linked["principal"] = owner
    pid = _avp.create_policy(policyStoreId=_STORE, definition={"templateLinked": linked})["policyId"]
    note(pid, kind, token, title, url, by)
    logger.info("protection added: by=%s scope=%s %s %s policy=%s",
                by, owner["entityId"] if owner else "everyone", kind, token, pid)
    return pid


def unprotect(owner: dict | None, policy_id: str, by: str) -> bool:
    """Delete a rule, only if it is this owner's (or, owner None, an admin rule). The stored
    principal decides — a request naming someone else's policy id deletes nothing."""
    try:
        p = _avp.get_policy(policyStoreId=_STORE, policyId=policy_id)
    except _avp.exceptions.ResourceNotFoundException:
        return False
    linked = (p.get("definition") or {}).get("templateLinked") or {}
    if linked.get("policyTemplateId") != _template(owner) or (owner and linked.get("principal") != owner):
        logger.warning("protection delete refused: by=%s policy=%s is not theirs", by, policy_id)
        return False
    _avp.delete_policy(policyStoreId=_STORE, policyId=policy_id)
    _notes.delete_item(Key=_note_key(policy_id))
    logger.info("protection removed: by=%s scope=%s policy=%s",
                by, owner["entityId"] if owner else "everyone", policy_id)
    return True
