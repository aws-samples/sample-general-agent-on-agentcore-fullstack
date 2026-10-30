"""Per-user grants to downstream systems (Lark, Google, GitHub) through AgentCore Identity:
which system each tool acts on, the consent wall a tool without a grant stops at, the consent
link, and what /auth and whoami report as granted, as whom.
"""

from __future__ import annotations

import json
import logging
import os

# At module level: the tool's `config: RunnableConfig` is resolved here for injection.
from langchain_core.runnables import RunnableConfig

from identity import lark_3lo

log = logging.getLogger("agent.grants")

# The second downstream system. Empty unless Google is configured; the agent then simply has
# no Google tools, the same way it has no search without a gateway.
_GOOGLE_MCP_URL = os.environ.get("GOOGLE_MCP_URL", "")


_GOOGLE_PROVIDER = os.environ.get("GOOGLE_OAUTH_PROVIDER", "")


_GOOGLE_SCOPES = os.environ.get(
    "GOOGLE_SCOPES", "openid email profile https://www.googleapis.com/auth/calendar.readonly").split()


# GitHub, through GitHub's own remote MCP server rather than one of ours: it takes the user's
# OAuth-App token as a bearer (measured 2026-10-26), so there is no Runtime to run. Read-only
# and a few toolsets by default — a workshop needs to read code, not to push it.
_GITHUB_PROVIDER = os.environ.get("GITHUB_OAUTH_PROVIDER", "")


_GITHUB_MCP_URL = os.environ.get("GITHUB_MCP_URL", "https://api.githubcopilot.com/mcp/")


_GITHUB_SCOPES = os.environ.get("GITHUB_SCOPES", "repo read:user").split()


_GITHUB_TOOLSETS = os.environ.get("GITHUB_MCP_TOOLSETS", "context,repos,issues,pull_requests,users")


_GITHUB_READONLY = os.environ.get("GITHUB_MCP_READONLY", "true").lower() != "false"


# Systems besides Lark, keyed like the router's IDP_REGISTRY → (MCP url, provider, scopes).
# Only configured ones appear, so the rest of the code never asks "is Google set up?".
_DOWNSTREAMS = {k: v for k, v in {
    "google": (_GOOGLE_MCP_URL, _GOOGLE_PROVIDER, _GOOGLE_SCOPES),
    "github": (_GITHUB_MCP_URL, _GITHUB_PROVIDER, _GITHUB_SCOPES),
}.items() if v[0] and v[1]}


def _github_connection(token: str, url: str) -> dict:
    """GitHub's remote MCP server, called with the user's own token."""
    from datetime import timedelta
    headers = {"Authorization": f"Bearer {token}", "X-MCP-Toolsets": _GITHUB_TOOLSETS}
    if _GITHUB_READONLY:
        headers["X-MCP-Readonly"] = "true"
    return {"transport": "streamable_http", "url": url, "headers": headers,
            "timeout": timedelta(seconds=60)}


def _github_refuses(token: str) -> bool:
    """Whether GitHub turns this token away (401: revoked on GitHub's side, or expired). An
    outage is not a refusal — that grant still stands."""
    import httpx
    try:
        return httpx.get("https://api.github.com/user", timeout=5,
                         headers={"Authorization": f"Bearer {token}"}).status_code == 401
    except Exception:  # noqa: BLE001
        return False


# A vendor's own server, reached with the user's token → (connection for a token, whether the
# vendor refuses a token). It lists nothing without a good token, so until there is one the
# session carries a stand-in that walls on first use instead.
_VENDOR_MCP = {"github": (_github_connection, _github_refuses)}


# How each system is named to the user; capitalize() would say "Github".
_LABELS = {"lark": "Lark", "google": "Google", "github": "GitHub"}


def _label(idp: str) -> str:
    return _LABELS.get(idp, idp.capitalize())


def _needs_grant_tool(idp: str):
    """The one tool a system whose server lists nothing without a grant contributes until the
    user grants it. Calling it stops the turn at the consent wall, like any of that system's
    tools; once granted, the session is rebuilt with the server's real tools."""
    from langchain_core.tools import StructuredTool
    label = _label(idp)
    return StructuredTool.from_function(
        func=lambda: _NEEDS_TOKEN_MARKER, name=idp,
        description=(  # nosemgrep: string-concat-in-list — one wrapped description, not two items
            f"Work with the user's {label}: repositories, files, commits, issues, pull "
            f"requests, releases. Call it before anything {label} — the user has not "
            f"authorised {label} yet, so this asks them to; afterwards the {label} tools "
            "themselves are here."))


_AUTH_PROMPT = (
    "To do that I need access to your {label} account. Please authorize once here, "
    "then send your message again:\n{url}"
)


# Every MCP server's reply when a tools/call arrives without a user token. It shows up as a
# tool result, not necessarily in the model's final reply — the model may paraphrase or
# translate. Scanning the tool results is what makes this robust.
_NEEDS_TOKEN_MARKER = "no user token (authorize first)"


def _walled_idp(session: dict, tool_name: str) -> str | None:
    """The system `tool_name` acts on, if this session holds no grant for it."""
    idp = (session.get("tool_idp") or {}).get(tool_name)
    return idp if idp in (session.get("consent") or {}) else None


def _hit_auth_wall(session: dict) -> str | None:
    """The system this turn needs consent for, or None — either because the stream was
    aborted at one of its tool calls (the fast path, before the model can editorialise) or
    because a tool result carried the server's refusal (the fallback, e.g. the sync path)."""
    tool = session.pop("walled_tool", None)
    if tool:
        return _walled_idp(session, tool)
    return _hit_auth_wall_from_tool_results(session)


def _hit_auth_wall_from_tool_results(session: dict) -> str | None:
    """The system whose server refused a call this turn for want of a user token.

    Reads the tool results collected while streaming rather than the model's answer,
    because the model paraphrases errors — 'no user token is available' would slip
    through a string check on the reply, and did in end-to-end testing."""
    for name, text in session.get("tool_texts") or []:
        if _NEEDS_TOKEN_MARKER in text:
            idp = _walled_idp(session, name)
            if idp:
                return idp
    return None


def _consent_link(session: dict, idp: str, workload_token: str = "") -> str:
    """The link to consent to `idp`, minted now if the build deferred it. "" when the vault
    turns out to hold a grant after all (its reads are nondeterministic, see _get_session)."""
    link = session["consent"].get(idp)
    if link or idp not in _DOWNSTREAMS:
        return link or ""
    _, provider, scopes = _DOWNSTREAMS[idp]
    kind, value = lark_3lo.get_user_token_for(
        session["actor_id"], provider, scopes, workload_token,
        force=idp in session.get("refused", ()))
    if kind != "auth_url":
        return ""
    session["consent"][idp] = value
    return value


def auth_status(actor_id: str, workload_token: str = "") -> dict:
    """Which IdPs this user has authorised → {"lark": bool, "google": bool, …, "accounts":
    {idp: who signed in to grant it}} — the account so a user can tell which of theirs it is.

    Only the agent can answer this. A consent is vaulted against the workload identity the
    Runtime derives from the inbound JWT, and re-deriving one from a fresh JWT for the same
    user reads a different namespace — see docs/agentcore-behavior.md."""
    status, tokens = {}, {}
    # Lark only for a Lark user of a deployment that has it; a failed lookup is "not
    # authorised", and must not keep the other systems from being reported.
    if actor_id.startswith("lark:") and lark_3lo.configured():
        try:
            kind, value = lark_3lo.get_user_lark_token(actor_id, workload_token=workload_token)
        except Exception:  # noqa: BLE001
            log.warning("Lark status lookup failed for %s", actor_id, exc_info=True)
            kind, value = "error", ""
        if kind == "token":
            tokens["lark"] = value
        status["lark"] = kind == "token"
    elif lark_3lo.configured():
        provider, scopes = lark_3lo.lark_provider()
        tokens["lark"] = lark_3lo.peek_user_token_for(actor_id, provider, scopes, workload_token)
        status["lark"] = bool(tokens["lark"])
    for idp, (_, provider, scopes) in _DOWNSTREAMS.items():
        tokens[idp] = lark_3lo.peek_user_token_for(actor_id, provider, scopes, workload_token)
        status[idp] = bool(tokens[idp])
    # Asked of each vendor at once: a round trip apiece, and /auth waits for the slowest.
    from concurrent.futures import ThreadPoolExecutor
    held = {idp: token for idp, token in tokens.items() if token and idp in _ACCOUNT_OF}
    with ThreadPoolExecutor(max_workers=max(1, 2 * len(held))) as pool:
        found = {idp: pool.submit(_ACCOUNT_OF[idp], token) for idp, token in held.items()}
        # A grant the vendor turns away is not one: /auth <idp> must offer a consent for it.
        refused = {idp: pool.submit(_VENDOR_MCP[idp][1], token)
                   for idp, token in held.items() if idp in _VENDOR_MCP}
    status.update((idp, False) for idp, f in refused.items() if f.result())
    status["accounts"] = {idp: f.result() for idp, f in found.items()
                          if f.result() and status.get(idp)}
    return status


def lark_account(actor_id: str, workload_token: str = "", resources: list | None = None) -> dict:
    """The Lark open_id this user's vaulted grant belongs to → {"open_id", "titles"}; "" when
    there is none. `resources` ([{kind, token}]) also asks Lark, with the same grant, for each
    one's title and link, which is how a protection rule is shown to the person maintaining it.

    What protection rules are keyed by (lambda/router/protection.py): the router cannot read the
    vault, and a web user's grant may be any Lark account they signed in with, so it is read
    from Lark with the grant itself. Never starts a consent."""
    if not lark_3lo.configured():
        return {"open_id": "", "titles": {}}
    if actor_id.startswith("lark:"):
        kind, token = lark_3lo.get_user_lark_token(actor_id, workload_token=workload_token)
        token = token if kind == "token" else ""
    else:
        provider, scopes = lark_3lo.lark_provider()
        token = lark_3lo.peek_user_token_for(actor_id, provider, scopes, workload_token)
    if not token:
        return {"open_id": "", "titles": {}}
    return {"open_id": lark_3lo._token_owner(token),
            "titles": lark_3lo.titles_of(token, resources) if resources else {}}


def _whoami_tool(actor_id: str, workload_token: str):
    """Who the person in this conversation is, for the model to answer from rather than
    guess: how they signed in, as whom, and which systems they have authorised as whom. The
    account comes from the run (the verified token's), not the session: a session may have
    been built by a request that carried none."""
    from langchain_core.tools import StructuredTool

    def whoami(config: RunnableConfig = None) -> str:
        account = ((config or {}).get("configurable") or {}).get("account") or ""
        web = actor_id.startswith("web:")
        try:
            status = auth_status(actor_id, workload_token)
        except Exception:  # noqa: BLE001 — who they are still answers without the grants
            log.warning("whoami could not read the grants", exc_info=True)
            status = {}
        accounts = status.get("accounts") or {}
        systems = [k for k in status if k != "accounts"]
        lark_name = accounts.get("lark", "") if not web else ""
        return json.dumps({
            "signed_in_with": "the web page (its own sign-in)" if web else "Lark",
            "account": account if web else (lark_name or actor_id.removeprefix("lark:")),
            "user_id": actor_id,
            "authorised": {k: accounts.get(k) or "yes, account unknown"
                           for k in systems if status.get(k)},
            "not_authorised": [k for k in systems if not status.get(k)],
        }, ensure_ascii=False)

    return StructuredTool.from_function(
        func=whoami, name="whoami",
        description=(  # nosemgrep: string-concat-in-list — one wrapped description, not two items
            "Who the person you are talking to is: how they signed in, their account, and "
            "which systems (Lark, Google, GitHub) they have authorised and with which account. Call "
            "it instead of guessing who they are; a system they have not authorised is "
            "authorised from the chat when a tool needs it."))


def _google_account(token: str) -> str:
    """The email of the Google account that granted this token; "" if it can't be read."""
    import httpx
    try:
        r = httpx.get("https://openidconnect.googleapis.com/v1/userinfo",
                      headers={"Authorization": f"Bearer {token}"}, timeout=5)
        return (r.json() or {}).get("email") or ""
    except Exception as e:  # noqa: BLE001 — a missing name must not cost the status
        log.warning("could not read the Google account: %s", type(e).__name__)
        return ""


def _github_account(token: str) -> str:
    """The login of the GitHub account that granted this token; "" if it can't be read."""
    import httpx
    try:
        r = httpx.get("https://api.github.com/user", timeout=5,
                      headers={"Authorization": f"Bearer {token}",
                               "Accept": "application/vnd.github+json"})
        return (r.json() or {}).get("login") or ""
    except Exception as e:  # noqa: BLE001 — a missing name must not cost the status
        log.warning("could not read the GitHub account: %s", type(e).__name__)
        return ""


# Who signed in to grant each system's token, read with that token: shown by /auth.
_ACCOUNT_OF = {"lark": lark_3lo.account_of, "google": _google_account, "github": _github_account}
