"""The tools a user's session gets, and where each component plugs in: the user's downstream
systems (Lark, Google, GitHub — identity/grants.py), web search through the AgentCore Gateway
(web_search.py), long-term memory in AgentCore Memory (memory.py), code execution in the Code
Interpreter (code_interpreter.py), and whoami.

Per-user Lark access (agent-side 3LO): for each end-user the agent fetches that user's
vaulted Lark token from AgentCore Identity (GetResourceOauth2Token, USER_FEDERATION)
and connects directly to the lark-mcp Runtime, passing the token in a custom header
(the server reads it and calls Lark as that user). The token gates tool *calls*, not
the connection — lark-mcp lists its tools without one — so an unauthorized user gets a
working session and is only asked to consent if the model actually reaches for a Lark
tool. 3LO is agent-side because a Runtime-hosted MCP server cannot be handed a per-user
token by the Gateway at all (see docs/agentcore-behavior.md).
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack

from langchain_mcp_adapters.sessions import create_session
from langchain_mcp_adapters.tools import load_mcp_tools

from tools import code_interpreter
from identity import lark_3lo
from tools import memory
from runtime import telemetry
from tools import web_search
from identity import grants

log = logging.getLogger("agent.tools")

# Empty unless ./deploy.sh mcp approval ran — the approval tools are opt-in.
_APPROVAL_MCP_URL = os.environ.get("APPROVAL_MCP_URL", "")


def _long_term_tools(actor_id: str) -> list:
    """Long-term memory reaches the model as two tools (remember/recall), not as a Store.

    A LangGraph Store would write every exchange as conversational events — the
    by-product pattern ADR 0008 rejected, and the one that makes retrieval cost scale with
    chat volume rather than with need. Nothing reads a Store here now that /status counts
    checkpoints instead of events."""
    return memory.tools_for(actor_id)


async def _aopen_tools(stack: AsyncExitStack, connection: dict) -> list:
    """Open an MCP session that stays alive for the session's lifetime and return its
    tools. The stack owns the teardown, so one close() releases every server."""
    session = await stack.enter_async_context(create_session(connection))
    await session.initialize()
    return await load_mcp_tools(session)


async def assemble(actor_id: str, stack: AsyncExitStack, workload_token: str = "") -> dict:
    """The tools this user's session gets, and what the consent wall needs to know
    about them. Agent-side 3LO: fetch the user's vaulted Lark
    token if there is one and open MCP sessions to the tool servers (SigV4 + the token
    in the custom header), kept open on `stack`, which the session owns.

    The token gates tool *calls*, not the connection: lark-mcp answers initialize and
    tools/list without it and only rejects tools/call with "authorize first" (verified
    against the deployed Runtime — an empty header returns the full tool list). So an
    unauthorized user still gets a session with the Lark tools listed, and can chat
    freely; consent is asked for when a tool is actually reached, which is the point at
    which the user can see what it is for."""
    tools: list = []
    auth_url = None
    identity_error = None
    # Consent is per system: each tool is tagged with the system whose server listed it, and
    # a call is walled only when *that* system has no grant. Untagged tools never wall.
    tool_idp: dict[str, str] = {}
    consent: dict[str, str] = {}   # idp → consent link; "" until a tool of it is reached
    refused: set[str] = set()      # systems whose vaulted grant the vendor turns away
    # Where each tool came from, for /tools. Tagged where a group is added rather than per
    # tool, so a server gaining or losing a tool needs no change here. Not the same as
    # tool_idp: the approval tools are gated by the Lark grant but are their own group.
    tool_source: dict[str, str] = {}

    def _tag(opened: list, source: str) -> list:
        tool_source.update((t.name, source) for t in opened)
        return opened

    async def _open(connection: dict, idp: str = "", source: str = "") -> list:
        opened = await _aopen_tools(stack, connection)
        if idp:
            tool_idp.update((t.name, idp) for t in opened)
        return _tag(opened, source or idp)
    # A Lark user's Lark grant must be their own Lark account, which only the Lark path can
    # check (it asks Lark whose token it is). A web user has no Lark identity to match: for
    # them Lark is one more system, granted on demand like Google, by whichever account they
    # consent with — and, like Google's, read without minting a link at build.
    lark_user = actor_id.startswith("lark:") and lark_3lo.configured()
    lark_web = not actor_id.startswith("lark:") and lark_3lo.configured()
    kind, value = "none", ""
    if lark_web:
        try:
            provider, scopes = lark_3lo.lark_provider()
            value = await asyncio.to_thread(
                lark_3lo.peek_user_token_for, actor_id, provider, scopes, workload_token)
        except Exception:  # noqa: BLE001 — chat without Lark tools beats no reply
            log.exception("Lark grant lookup failed for %s", actor_id)
        kind = "token" if value else "none"
        if not value:
            consent["lark"] = ""     # minted when the user asks to authorise
    if lark_user:
        try:
            kind, value = await asyncio.to_thread(
                lark_3lo.get_user_lark_token, actor_id, False, workload_token)
        except Exception as e:  # noqa: BLE001 — never crash the turn on identity hiccups
            log.exception("3LO token lookup failed for %s", actor_id)  # nosemgrep: python-logger-credential-disclosure — logs ids and codes, never values
            kind, value = "error", str(e)

    if kind == "auth_url":
        auth_url = value  # remembered for the tool-call path, not returned upfront
        consent["lark"] = value
    elif kind == "wrong_owner":
        telemetry.count("identity_refused", reason="wrong_owner")
        # A claimed actor that does not own the vaulted token. Refused rather than offered a
        # consent link: minting one would burn the real owner's grant.
        identity_error = "这个身份与授权的账号不一致，无法代表其操作。"
    elif kind not in ("token", "none"):
        # Surface identity failures instead of running tool-less: an agent that merely
        # says "I have no tools" reads as model behaviour and hides the real cause (a
        # missing IAM permission, most likely).
        identity_error = value

    token = value if kind == "token" else ""

    if (lark_user or lark_web) and identity_error is None:
        try:
            tools += await _open(lark_3lo.mcp_connection_for(token), "lark")
        except Exception:  # noqa: BLE001 — chat without Lark tools beats no reply
            log.exception("lark-mcp unavailable for %s", actor_id)

        # Approval tools, when that Runtime is deployed. Separate server because its
        # write operations run on the app's tenant token (Lark accepts no user token
        # there) and its decision limits have to be bound to specific tools — see
        # docs/architecture.md. The user token is still passed: add_sign needs it.
        if _APPROVAL_MCP_URL:
            try:
                tools += await _open(
                    lark_3lo.mcp_connection_for(token, url=_APPROVAL_MCP_URL), "lark",
                    source="approval")
            except Exception:  # noqa: BLE001 — optional, like search
                log.exception("approval tools unavailable for %s", actor_id)

    # Other systems, independent of the Lark grant: a user may have consented to one and not
    # the other. Tools are listed either way (servers gate calls, not discovery); a missing
    # grant only records the system, and its link is minted if one of its tools is reached —
    # minting here would start a consent flow nobody asked for, killing any link in flight.
    for idp, (url, provider, scopes) in grants._DOWNSTREAMS.items():
        try:
            d_token = await asyncio.to_thread(
                lark_3lo.peek_user_token_for, actor_id, provider, scopes, workload_token)
            vendor = grants._VENDOR_MCP.get(idp)
            if vendor and d_token and await asyncio.to_thread(vendor[1], d_token):
                # The vault still holds a grant the vendor no longer honours: ask again, and
                # force past it — an ordinary consent would find it and say "already granted".
                d_token = ""
                refused.add(idp)
            if not d_token:
                consent[idp] = ""
            if vendor and not d_token:
                stand_in = grants._needs_grant_tool(idp)
                tool_idp[stand_in.name] = idp
                tools += _tag([stand_in], idp)
            elif vendor:
                tools += await _open(vendor[0](d_token, url), idp)
            else:
                tools += await _open(lark_3lo.mcp_connection_for(d_token, url=url), idp)
        except Exception:  # noqa: BLE001 — optional, like search
            log.exception("%s tools unavailable for %s", idp, actor_id)

    # Search doesn't depend on the user's Lark grant, so add it even when the Lark
    # tools are unavailable — an unauthorised user can still ask questions.
    if web_search.available():
        try:
            conn = await asyncio.to_thread(web_search.connection_for, actor_id)
            tools += _tag(await _aopen_tools(stack, conn), "web_search")
        except Exception:  # noqa: BLE001 — search is optional, Lark tools are not
            log.exception("web search unavailable for %s", actor_id)

    tools += _tag(_long_term_tools(actor_id), "memory")
    tools += _tag([grants._whoami_tool(actor_id, workload_token)], "identity")

    # Code execution, when an interpreter is deployed. The sandbox is created here rather
    # than per call so one conversation keeps one workspace session — and it is kept on the
    # cached entry so _close_session can stop it. Nothing is started yet: the session only
    # begins when the model first runs something.
    sandbox = code_interpreter._Sandbox(actor_id) if code_interpreter.available() else None
    if sandbox is not None:
        tools += _tag(code_interpreter.tools_for(actor_id, sandbox), "code")
    return {"tools": tools, "sandbox": sandbox,
            "auth_url": auth_url, "identity_error": identity_error,
            "consent": consent, "refused": refused, "tool_idp": tool_idp,
            "tool_source": tool_source}
