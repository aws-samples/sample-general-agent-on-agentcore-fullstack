"""AG-UI on the same Runtime, so a browser can stream a turn without a backend in between.

AgentCore proxies a streamed response under `serverProtocol=HTTP` (measured: chunks arrive
as the container writes them, `text/event-stream` passes through), so the AG-UI contract is
served from the existing agent Runtime rather than a second one with `serverProtocol=AGUI`.
One runtime, one image, one deployment. See docs/agentcore-behavior.md.

The browser calls `/invocations` directly with its Cognito JWT as Bearer. server.py has
verified it and put the actor it names on the payload before this module sees the request.
"""

from __future__ import annotations

import json
import logging
import time

from aiohttp import web

import agent_core

log = logging.getLogger("agent.agui")

# AG-UI's own request shape. `action` marks the router's JSON protocol instead, so the two
# share `/invocations` without ambiguity.
_AGUI_KEYS = ("threadId", "messages", "runId")


def is_agui_request(payload: dict) -> bool:
    return "action" not in payload and any(k in payload for k in _AGUI_KEYS)


def _sse(event) -> bytes:
    """One AG-UI event as an SSE frame. Pydantic models serialise by alias, which is what
    the protocol's camelCase field names require."""
    body = event.model_dump_json(by_alias=True, exclude_none=True)
    return f"data: {body}\n\n".encode()


async def _error_stream(request: web.Request, message: str) -> web.StreamResponse:
    """A RUN_ERROR stream. HTTP 200 on purpose: the AG-UI contract distinguishes a
    connection-level failure (non-2xx, before the stream) from an agent-level one, which
    travels as an event so the client can render it in the conversation."""
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                       "Cache-Control": "no-cache"})
    await resp.prepare(request)
    await resp.write(f'data: {json.dumps({"type": "RUN_ERROR", "message": message})}\n\n'
                     .encode())
    await resp.write_eof()
    return resp


def _run_input_fields(payload: dict, thread_id: str) -> dict:
    """The page's RunAgentInput with our own fields taken out and the thread replaced: AG-UI's
    threadId is the page's to choose, and it decides whose conversation this is."""
    fields = {k: v for k, v in payload.items() if k not in ("actorId", "memorySessionId")}
    return {**fields, "threadId": thread_id, "runId": payload.get("runId") or thread_id}


def _consent(session: dict) -> bytes:
    """A CUSTOM `consent` event naming the system a tool of this run was refused for, else
    b"". No link in it: the page mints one through `/auth` when the user clicks, with the
    token it will complete with — only the token a consent was started with can complete it,
    and a link minted now could be killed by a later build before it is clicked."""
    try:
        idp = agent_core._hit_auth_wall(session)
    except Exception:  # noqa: BLE001 — a missing prompt must not cost the answer
        log.warning("could not check for a consent wall", exc_info=True)
        return b""
    if not idp:
        return b""
    body = {"type": "CUSTOM", "name": "consent", "value": {"idp": idp}}
    return f"data: {json.dumps(body)}\n\n".encode()


def _walls(session: dict, event) -> bool:
    """Whether `event` starts a tool whose system has no grant. Such a call is certain to be
    refused, so the run stops there, as a Lark turn does, rather than have the model narrate
    the refusal before the page offers to authorise."""
    if getattr(event, "type", "") != "TOOL_CALL_START":
        return False
    name = getattr(event, "tool_call_name", "") or ""
    if not agent_core._walled_idp(session, name):
        return False
    session["walled_tool"] = name
    log.info("aborting web turn: %s needs consent", name)
    return True


async def handle(request: web.Request, payload: dict,
                 workload_token: str) -> web.StreamResponse:
    """Run one AG-UI turn, streaming events as they are produced."""
    received_ms = int(time.time() * 1000)   # the user's message is stamped with this
    # Verified by server.py from the bearer, not taken from what the page sent.
    actor_id = payload.get("actorId") or ""
    if not actor_id:
        return await _error_stream(request, "请求缺少身份信息，请重新打开页面。")

    try:
        from ag_ui.core.types import RunAgentInput
        from ag_ui_langgraph import LangGraphAgent
    except Exception:  # noqa: BLE001 — absent-not-broken, like every optional surface here
        log.exception("ag-ui packages unavailable")
        return await _error_stream(request, "Web chat is not available in this build.")

    # The session carries the tools, the checkpointer and the summarisation middleware, so a
    # web turn lands in the same conversation as a Lark turn for the same person.
    # The thread the page names was checked against the actor by server.py; a page that
    # names none gets the one the router last named.
    session = await agent_core.aget_session(actor_id, workload_token=workload_token,
                                            thread=payload.get("memorySessionId") or "")
    # Only a Lark user's own Lark lookup can fail this way; a system not yet granted is not
    # an error here — its consent link comes at the end of the run that needs it.
    if session.get("identity_error"):
        return await _error_stream(request, "暂时无法访问你的 Lark 账号，请稍后再试。")

    # One turn per thread: two would each read the other's half-written checkpoint. A turn
    # already running elsewhere would answer this in that window, so the page is told rather
    # than have its message steered there. Claimed in one step, so two tabs cannot both pass.
    thread_id = session["mem_sid"]
    owner = agent_core.try_claim(thread_id, "web")
    if owner:
        return await _error_stream(request, agent_core._BUSY_ELSEWHERE.get(
            owner, agent_core._BUSY_ELSEWHERE["sync"]))

    # Set by the run itself when it stops at a tool that needs a grant.
    session.pop("walled_tool", None)
    # From the claim until the run has started, a failure — a malformed payload included —
    # has to release the thread itself: no run exists yet to release it.
    try:
        # thread_id comes from the session, never from the request: it decides whose
        # conversation this is.
        run_input = RunAgentInput(**_run_input_fields(payload, thread_id))
        # The page has no way to carry a file yet, and this session is also served to
        # Lark, so the run says so explicitly rather than inheriting whatever chat was
        # served last.
        agui = LangGraphAgent(name="agentcore-fullstack", graph=session["graph"],
                              config=agent_core._run_config(
                                  session, agent_core.delivery.NoDelivery("the web page"),
                                  received_ms))
        # The graph's MCP sessions live on agent_core's own loop, so the run happens there
        # and the events come back over a queue. Released when the run finishes, not when
        # this handler stops reading: a page closed mid-answer leaves the graph running,
        # and a Lark message must see the thread busy until it ends.
        events = agent_core.stream_on_agent_loop(
            lambda: agui.run(run_input), on_done=lambda: agent_core.release(thread_id),
            until=lambda event: _walls(session, event))
    except BaseException:
        agent_core.release(thread_id)
        raise

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                       "Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})
    await resp.prepare(request)
    # Tool results, for the consent check the Lark path makes from the same texts.
    session["tool_texts"] = []
    names: dict[str, str] = {}
    try:
        async for event in events:
            kind = getattr(event, "type", "")
            if kind == "TOOL_CALL_START":
                names[event.tool_call_id] = event.tool_call_name
            elif kind == "TOOL_CALL_RESULT":
                session["tool_texts"].append((names.get(event.tool_call_id, ""),
                                              str(getattr(event, "content", "") or "")))
            elif kind == "RUN_FINISHED":
                # Before the run ends, or a client that stops at RUN_FINISHED never sees it.
                prompt = _consent(session)
                if prompt:
                    await resp.write(prompt)
            await resp.write(_sse(event))
        # Stopped at a tool that needs a grant: the run never finished by itself.
        if session.get("walled_tool"):
            finished = {"type": "RUN_FINISHED", "threadId": thread_id, "runId": run_input.run_id}
            await resp.write(_consent(session) + f"data: {json.dumps(finished)}\n\n".encode())
    except Exception as e:  # noqa: BLE001 — the stream is open, so report in-band
        log.exception("agui turn failed for %s", actor_id)
        await resp.write(f'data: {json.dumps({"type": "RUN_ERROR", "message": type(e).__name__})}\n\n'
                         .encode())
    finally:
        await resp.write_eof()
    return resp
