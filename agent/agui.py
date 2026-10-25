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
import uuid

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
    await resp.write(f'data: {json.dumps({"type": "RUN_ERROR", "message": message}, ensure_ascii=False)}\n\n'
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


def _custom(event: dict) -> bytes:
    """{name, value} as a CUSTOM event frame."""
    return f"data: {json.dumps({'type': 'CUSTOM', **event})}\n\n".encode()


class _Mark:
    """A CUSTOM event made inside the run, carried through the run's own event queue."""

    def __init__(self, body: dict) -> None:
        self.body = body


class _PageMarks:
    """Stands in for the Lark card on a web turn (agent_core._turn_cards): what happens inside
    the graph that the token stream cannot show, put on the run's own stream."""

    def __init__(self, out: list) -> None:
        self.out = out

    def steered(self, _reply_to: str = "") -> None:
        """What follows answers a message sent mid-turn, so the page puts it below that."""
        self.out.append({"name": "steered", "value": {}})

    def compacting(self, started: bool) -> None:
        self.out.append({"name": "compacting", "value": {"started": bool(started)}})


def _end(thread_id: str, marks: _PageMarks, state: dict) -> list:
    """The run is over: drop its marks and, unless the run already released the thread in
    the same step as finding nothing left to answer, release it — exactly once, since a
    second release would free a claim another turn may have taken meanwhile. Messages still
    queued were acknowledged to the page, so they are reported, not dropped."""
    if agent_core._turn_cards.get(thread_id) is marks:
        agent_core._turn_cards.pop(thread_id, None)
    if state.get("released"):
        return []
    state["released"] = True
    leftover = agent_core.release(thread_id)
    if not leftover:
        return []
    agent_core.telemetry.count("steer_unanswered", n=str(len(leftover)))
    return [_Mark({"name": "unanswered", "value": {"count": len(leftover)}})]


def _steerable(payload: dict) -> tuple[str, str]:
    """The text and id of the page's message, if it can join a running turn: text only —
    a picture on the web travels inline in the message, which a steer cannot carry."""
    messages = payload.get("messages") or []
    last = messages[-1] if messages else {}
    if not isinstance(last, dict) or last.get("role") != "user":
        return "", ""
    content = last.get("content")
    if isinstance(content, str) and content.strip():
        return content, str(last.get("id") or "")
    return "", ""


def _pictures_handed_over(payload: dict, thread_id: str) -> dict:
    """The page's message with its pictures moved into the turn's registry, as a Lark
    message's are: the conversation keeps a placeholder, the pixels reach this turn's model
    calls only (bound to the message id), and nothing is re-billed on later turns."""
    messages = payload.get("messages") or []
    last = messages[-1] if messages else None
    if not isinstance(last, dict) or not isinstance(last.get("content"), list):
        return payload
    text, handles, blocks = agent_core.images.from_page(last["content"])
    if not handles:
        return payload
    marks, message_id = agent_core.images.register(thread_id, handles, blocks)
    content = "\n".join(t for t in (text or "看看这张图。", marks) if t)
    return {**payload, "messages": [*messages[:-1],
                                    {**last, "id": message_id, "content": content}]}


def _stream_of(payload: dict) -> str:
    """Which page's stream a request comes from — one per page load, so a reloaded tab, a
    second tab and another device are each a destination of their own. Not a credential:
    every one of them is the same verified actor; it only says where an answer would show."""
    props = payload.get("forwardedProps") or {}
    stream = props.get("streamId") if isinstance(props, dict) else ""
    return f"web:{stream}" if isinstance(stream, str) and stream else ""


async def _steered_stream(request: web.Request, thread_id: str, run_id: str) -> web.StreamResponse:
    """The answer to a message handed to the running turn: that turn's stream carries the
    reply, so this one only says it was taken — as a complete, if empty, AG-UI run."""
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                       "Cache-Control": "no-cache"})
    await resp.prepare(request)
    ids = {"threadId": thread_id, "runId": run_id}
    await resp.write(f'data: {json.dumps({"type": "RUN_STARTED", **ids})}\n\n'.encode()
                     + _custom({"name": "queued", "value": {}})
                     + f'data: {json.dumps({"type": "RUN_FINISHED", **ids})}\n\n'.encode())
    await resp.write_eof()
    return resp


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

    # One turn per thread: two would each read the other's half-written checkpoint. A message
    # sent from the page whose turn is running joins that turn, as one sent in the same Lark
    # chat does; one from anywhere else would have its answer appear there, so it is told
    # instead. Claimed before the session is built, as chat_async does: a build may close a
    # session a running turn is using. The thread the page names was checked against the
    # actor by server.py; a page that names none gets the one the router last named.
    thread_id = payload.get("memorySessionId") or agent_core.thread_for(actor_id)
    dest = _stream_of(payload)
    text, message_id = _steerable(payload)
    owner = agent_core.claim_or_steer(thread_id, "web", dest, text, message_id) if text \
        else agent_core.try_claim(thread_id, "web", dest)
    if owner == "steered":
        return await _steered_stream(request, thread_id, payload.get("runId") or thread_id)
    if owner == "web":     # a picture, which cannot join a turn; is that turn this page's?
        owner = "picture" if dest and agent_core._claim_dest.get(thread_id) == dest \
            else "web-other"
    if owner == "picture":
        return await _error_stream(request, "这一轮回答结束后再发图片。")
    if owner:
        return await _error_stream(request, agent_core._BUSY_ELSEWHERE.get(
            owner, agent_core._BUSY_ELSEWHERE["sync"]))

    # What the graph does that the token stream cannot show — a file sent, a message steered
    # in, a summary being written — as CUSTOM events for this run's stream. Per run, never on
    # the session: that is shared with Lark, and a stored target is how a file reaches the
    # wrong surface.
    out: list = []
    marks = _PageMarks(out)
    state: dict = {}
    # From the claim until the run has started, a failure — a malformed payload or a failed
    # build included — has to release the thread itself: no run exists yet to release it.
    try:
        # The session carries the tools, the checkpointer and the summarisation middleware,
        # so a web turn lands in the same conversation as a Lark turn for the same person.
        session = await agent_core.aget_session(actor_id, workload_token=workload_token,
                                                thread=thread_id, own_claim=True)
        # Only a Lark user's own Lark lookup can fail this way; a system not yet granted is
        # not an error here — the run stops at the tool that needs it and the page asks.
        if session.get("identity_error"):
            _end(thread_id, marks, state)
            return await _error_stream(request, "暂时无法访问你的 Lark 账号，请稍后再试。")
        # Set by the run itself when it stops at a tool that needs a grant.
        session.pop("walled_tool", None)
        # thread_id was never taken from AG-UI's threadId: that one is the page's to choose.
        # Off the server's loop: decoding and downscaling a photo stalls every other stream.
        handed = await agent_core.arun_in_thread(_pictures_handed_over, payload, thread_id)
        run_input = RunAgentInput(**_run_input_fields(handed, thread_id))

        def agent_for(arrived_ms: int):
            return LangGraphAgent(name="agentcore-fullstack", graph=session["graph"],
                                  config=agent_core._run_config(
                                      session, agent_core.delivery.WebDelivery(out, actor_id),
                                      arrived_ms))

        async def turns():
            """The turn, then a continuation for whatever was steered in after its last
            model call — the loop chat_async runs for Lark — until nothing is left."""
            nonlocal session
            run_in, agent = run_input, agent_for(received_ms)
            arrived["ms"] = received_ms
            try:
                async for item in answer(run_in, agent):
                    yield item
            finally:
                # Stopped at a consent wall: the claim is still this run's until on_done.
                if not state.get("released"):
                    await stamp()

        arrived: dict = {}

        async def stamp() -> None:
            """A turn stopped before any reply still records when its message came — the
            arrival of the run's own message, a continuation's included. Only while the
            claim is held: after it, another turn may be writing the thread."""
            try:
                await agent_core.astamp_unanswered(session["graph"], session["config"],
                                                   arrived["ms"])
            except Exception:  # noqa: BLE001 — a missing time must not cost the release
                log.warning("could not stamp an unanswered message", exc_info=True)

        async def answer(run_in, agent):
            nonlocal session
            while True:
                failed = False
                async for event in agent.run(run_in):
                    failed = failed or getattr(event, "type", "") == "RUN_ERROR"
                    # Marks made while the graph produced this event go first, in the order
                    # they happened: the page opens a new card exactly where text resumes.
                    while out:
                        yield _Mark(out.pop(0))
                    yield event
                while out:
                    yield _Mark(out.pop(0))
                if failed:          # its message never got a reply; before the release
                    await stamp()
                run_in = None
                while run_in is None:
                    queued = agent_core.take_steer_or_release(thread_id)
                    if not queued:
                        state["released"] = True
                        return
                    marks.steered()
                    try:
                        # Fetched again, as chat_async does: a consent completed meanwhile
                        # drops the cached session, and its MCP connections with it.
                        session = await agent_core.aget_session(
                            actor_id, workload_token=workload_token, thread=thread_id,
                            own_claim=True)
                        arrived["ms"] = queued[0][3] if len(queued[0]) > 3 else 0
                        agent = agent_for(arrived["ms"])
                        run_in = RunAgentInput(**_run_input_fields({
                            "runId": f"{run_input.run_id}-{uuid.uuid4().hex[:8]}",
                            "messages": [{"id": q[1] or uuid.uuid4().hex, "role": "user",
                                          "content": q[0]} for q in queued],
                            "tools": [], "context": [], "state": {}, "forwardedProps": {}},
                            thread_id))
                    except Exception:  # noqa: BLE001 — off the queue already, so say so
                        log.exception("continuation failed for %s", actor_id)
                        yield _Mark({"name": "unanswered", "value": {"count": len(queued)}})

        agent_core._turn_cards[thread_id] = marks
        # The graph's MCP sessions live on agent_core's own loop, so the run happens there
        # and the events come back over a queue. Released when the run finishes, not when
        # this handler stops reading: a page closed mid-answer leaves the graph running,
        # and a Lark message must see the thread busy until it ends.
        events = agent_core.stream_on_agent_loop(
            turns, on_done=lambda: _end(thread_id, marks, state),
            until=lambda event: _walls(session, event))
    except BaseException:
        _end(thread_id, marks, state)
        raise

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                       "Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})
    await resp.prepare(request)
    names: dict[str, str] = {}
    try:
        async for event in events:
            if isinstance(event, _Mark):
                await resp.write(_custom(event.body))
                continue
            kind = getattr(event, "type", "")
            if kind == "RUN_STARTED":
                # Tool results, for the consent check the Lark path makes from the same
                # texts — per run, or a continuation would ask again for the same refusal.
                session["tool_texts"] = []
            elif kind == "TOOL_CALL_START":
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
