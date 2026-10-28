"""The Lark surface and the synchronous one: a Lark message answered into a streaming card
(chat_async), and a reply returned in the response (chat_result). The web page's surface is
agui.py.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from channels import delivery
from channels import images
from channels import lark_card
from runtime import telemetry
from identity import grants
from runtime import sessions
from runtime import turns

log = logging.getLogger("agent.lark")

def _release_unanswered(thread_id: str, chat_id: str) -> None:
    """Release after a turn that never ran, telling the chat about anything steered into it
    meanwhile — those messages were acknowledged, so dropping them silently would lose them."""
    leftover = turns.release(thread_id)
    if leftover:
        telemetry.count("steer_unanswered", n=str(len(leftover)))
        _quietly(lark_card.send_text, chat_id,
                 f"还有 {len(leftover)} 条消息没来得及处理，请重新发送。")


_IDENTITY_ERROR = (
    "I can't reach your Lark account right now, so I have no Lark tools this turn "
    "(other questions still work).\nDetails: {err}"
)


def chat_result(actor_id: str, message: str, email: str = "",
                mem_sid: str = "", workload_token: str = "") -> dict:
    """Non-streaming chat → {reply, needs_auth, auth_url}. History via Memory.
    When the user hasn't authorized Lark yet, needs_auth is True and auth_url is the
    raw consent URL, so the caller (router) can drive the wait-for-consent loop instead
    of asking the user to re-send."""
    received_ms = int(time.time() * 1000)   # the user's message is stamped with this
    sessions.remember_thread(actor_id, mem_sid)
    thread_id = mem_sid or sessions.thread_for(actor_id)
    # A turn is already running on this thread. This surface answers synchronously and the
    # running turn's answer appears on another surface, so steering the message there would
    # promise an answer this caller never receives. It is told to send again instead.
    owner = turns.try_claim(thread_id, "sync")
    if owner is not None:
        telemetry.count("busy_elsewhere", surface="sync")
        return {"reply": turns._BUSY_ELSEWHERE.get(owner, turns._BUSY_ELSEWHERE["sync"]),
                "needs_auth": False, "busy": owner}
    try:
        s = sessions._get_session(actor_id, email, thread_id, workload_token=workload_token,
                                  own_claim=True)
        if s.get("identity_error"):
            # Answer without tools, but say so — silently degrading is what made a missing
            # IAM permission look like the model choosing not to help.
            return {"reply": _IDENTITY_ERROR.format(err=s["identity_error"]),
                    "needs_auth": False, "identity_error": s["identity_error"]}
        s["tool_texts"] = []
        turn_id = "msg-" + uuid.uuid4().hex
        with telemetry.timed("turn", surface="sync"):
            state = turns._run(s["graph"].ainvoke(
                {"messages": [HumanMessage(message, id=turn_id)]},
                config=turns._run_config(s, delivery.NoDelivery("this channel"), received_ms)))
    finally:
        turns.release(thread_id)
    # This turn's tool results only. The state is the whole thread, so scanning all of it
    # let one old "no user token" result raise a consent prompt on every later sync turn.
    messages = state.get("messages", [])
    start = next((i for i, m in enumerate(messages) if getattr(m, "id", None) == turn_id), 0)
    for m in messages[start:]:
        if isinstance(m, ToolMessage):
            s["tool_texts"].append((m.name, _message_text(m)))
    reply = _message_text(state["messages"][-1]) if state.get("messages") else ""
    idp = grants._hit_auth_wall(s)
    link = grants._consent_link(s, idp, workload_token) if idp else ""
    if link:
        return {"reply": grants._AUTH_PROMPT.format(label=grants._label(idp), url=link),
                "needs_auth": True, "auth_url": link, "idp": idp}
    return {"reply": reply, "needs_auth": False}


def run_chat(actor_id: str, message: str, email: str = "",
             mem_sid: str = "") -> str:
    """Back-compat: assistant's final text (or the consent prompt)."""
    return chat_result(actor_id, message, email, mem_sid)["reply"]


def _quietly(fn, *args) -> None:
    """Run a best-effort side effect whose failure must not change what happens next."""
    try:
        fn(*args)
    except Exception:  # noqa: BLE001 — best-effort by definition
        log.warning("%s failed", getattr(fn, "__name__", fn), exc_info=True)


def chat_async(actor_id: str, message: str, chat_id: str, email: str = "",
               mem_sid: str = "", message_id: str = "", reaction_id: str = "",
               fresh_session: bool = False, workload_token: str = "",
               image_handles: list | None = None) -> dict:
    """Accept the work and answer later.

    A real task can outlast any request/response window (InvokeAgentRuntime caps at
    15 min, the calling Lambda at less), and a turn that times out mid-way is the worst
    outcome: the work often completed, but the user was told it failed. So we return an
    acknowledgement now and push the result to the chat when it's ready.

    An unauthorized user is not stopped here: the Lark tools are listed even without a
    token, so the turn runs and consent is only raised if the model actually calls
    one. By then the router has returned, so the prompt is pushed to the chat like any
    other answer and the user re-sends after approving."""
    received_ms = int(time.time() * 1000)   # the user's message is stamped with this
    sessions.remember_thread(actor_id, mem_sid)
    thread_id = mem_sid or sessions.thread_for(actor_id)
    handles = list(image_handles or [])
    blocks = images.prepare(handles)        # network, so before the lock

    # One step under the lock: see who holds the thread, then claim it, steer into it, or
    # step aside. A separate check and claim left a cold session build between them, so two
    # turns could start on one thread. On the same surface the message is steered into the
    # running turn, whose answer appears where the user is looking; on another surface it
    # would be answered in that window, so it is told instead — and registers nothing, so
    # it cannot join that turn's attachments.
    with turns._in_flight_lock:
        owner = turns._in_flight_threads.get(thread_id)
        # The running turn answers into another chat — a DM and a group share one thread.
        if owner == "lark" and turns._claim_dest.get(thread_id) != chat_id:
            owner = "lark-other"
        if owner in (None, "lark"):
            if owner is None:
                turns._track(+1, thread_id, "lark", chat_id)
            marks, turn_message_id = images.register(thread_id, handles, blocks)
            if marks:
                message = "\n".join([message, marks]).strip() if message else marks
            if owner == "lark":
                turns.steer(thread_id, message, turn_message_id, fresh=fresh_session)
                if message_id and not fresh_session:
                    turns._steer_reply_to[thread_id] = message_id
    if owner == "lark":
        telemetry.count("steered", surface="lark")
        _quietly(lark_card.remove_reaction, message_id, reaction_id)
        return {"accepted": True, "needs_auth": False, "steered": True}
    if owner is not None:
        telemetry.count("busy_elsewhere", surface="lark")
        _quietly(lark_card.remove_reaction, message_id, reaction_id)
        lark_card.send_text(chat_id, turns._BUSY_ELSEWHERE.get(
            owner, turns._BUSY_ELSEWHERE["sync"]))
        return {"accepted": True, "needs_auth": False, "busy": owner}

    # The thread is claimed from here on. Every way out has to release it, or the thread
    # reads as busy for good and every later message is steered into a turn that is gone.
    try:
        s = sessions._get_session(actor_id, email, thread_id, fresh=fresh_session,
                                  workload_token=workload_token, own_claim=True)
        if s.get("identity_error"):
            _release_unanswered(thread_id, chat_id)
            return {"reply": _IDENTITY_ERROR.format(err=s["identity_error"]),
                    "needs_auth": False, "identity_error": s["identity_error"]}

        def _answer(s: dict, inputs: list) -> None:
            try:
                with telemetry.timed("turn", surface="lark"):
                    _stream_to_chat(s, inputs, chat_id, received_ms=received_ms)
                idp = grants._hit_auth_wall(s)
                if idp:
                    telemetry.count("auth_wall", surface="lark", idp=idp)
                    link = grants._consent_link(s, idp, workload_token)
                    if link:
                        # The message was parked before this turn, so the shim's /return
                        # replays it once consent lands — no re-send.
                        lark_card.send_link(
                            chat_id, f"需要访问你的 {grants._label(idp)} 账号，授权后我会自动继续：",
                            "点击授权", link)
                    else:
                        sessions._drop_sessions(actor_id)
                        lark_card.send_text(chat_id, f"{grants._label(idp)} 已授权，请再发一次。")
            except Exception as e:  # noqa: BLE001 — the caller is already gone
                log.exception("async turn failed for %s", actor_id)
                # This session stays cached, so a turn that died mid tool call would make
                # every later turn fail too. Repair here as well as at build time.
                try:
                    turns._run(sessions._arepair_interrupted_turn(s["graph"], s["config"],
                                                                  own_claim=True),
                               timeout=30)
                except Exception:  # noqa: BLE001
                    log.warning("could not repair the interrupted turn", exc_info=True)
                _quietly(lark_card.send_text, chat_id,
                         f"Sorry, that didn't work out ({type(e).__name__}).")

        def _run_turn() -> None:
            session, inputs = s, [(message, turn_message_id, False, received_ms)]
            released = False
            try:
                while inputs:
                    _answer(session, inputs)
                    # A message steered in after the turn's last model call is never read
                    # by it — that window is the whole streamed answer, which is when people
                    # add "also this". So the claim is kept and the leftovers are answered as
                    # a continuation, with their attachments still registered; only an empty
                    # queue ends the turn, in the same step as releasing the thread.
                    with turns._in_flight_lock:
                        inputs = turns._take_steer(thread_id)
                        if not inputs:
                            turns.release(thread_id)
                            released = True
                    if inputs:
                        telemetry.count("steer_continued")
                        # Fetched again rather than reused: the consent branch above may
                        # have dropped this session, and a consent replay among the inputs
                        # needs one built after the grant, not the unauthorised one.
                        fresh = any(item[2] for item in inputs)
                        if fresh:
                            sessions._drop_sessions(actor_id)
                        # These inputs are already off the queue, so a failure here has to
                        # count them itself — the release at the end only sees what is left.
                        try:
                            session = sessions._get_session(
                                actor_id, email, thread_id, fresh=fresh,
                                workload_token=workload_token, own_claim=True)
                            problem = session.get("identity_error")
                        except Exception as e:  # noqa: BLE001 — told below, not raised
                            log.exception("continuation session failed for %s", actor_id)
                            problem = type(e).__name__
                        if problem:
                            _quietly(lark_card.send_text, chat_id,
                                     f"还有 {len(inputs)} 条消息没来得及处理（{problem}），请重新发送。")
                            break
            finally:
                if not released:
                    _release_unanswered(thread_id, chat_id)
                # Last, and on its own: it is a network call that can raise, and the claim
                # must be released whatever it does. A marker left behind is cosmetic.
                _quietly(lark_card.remove_reaction, message_id, reaction_id)

        threading.Thread(target=_run_turn, name=f"turn-{actor_id[:16]}", daemon=True).start()
    except BaseException:
        _release_unanswered(thread_id, chat_id)
        raise
    return {"accepted": True, "needs_auth": False}


# No throttle on the card write on purpose. The write is queued, not performed — the
# card's own worker coalesces to the newest text and is paced by Lark's round trip
# (~470 ms measured), so handing over every delta costs a lock and sets no rate. An
# earlier throttle guarded a blocking write from this loop, and at 0.4 s it was tuned
# below the write's real cost: the loop lost over half its time to HTTP and the text
# jerked.


def _message_text(msg) -> str:
    """Text of a LangChain message or chunk. Bedrock Converse returns content as a
    list of typed blocks, not a string, so both shapes have to be handled."""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


# Marks a queued call for the consumer to make, rather than text to yield.
_EVENT = object()


def _turn_input(message, message_id: str | None = None) -> list:
    """The HumanMessages a turn starts from: one string, or (text, id, fresh, ms) entries —
    several when a continuation answers what was steered in after the last model call."""
    if isinstance(message, str):
        return [HumanMessage(message, id=message_id)]
    return [turns._said(item) for item in message]


class _InStreamOrder:
    """A card's status calls, queued behind the text the stream produced before them. The
    graph runs on the agent loop and the card is fed from the consumer thread, so a call
    made straight from the loop could reach the card ahead of the last words before it.
    `current` is called when the event is handled: the card may have been replaced by then."""

    def __init__(self, events: queue.Queue, current, on_steered):
        self._events, self._current, self._on_steered = events, current, on_steered

    def compacting(self, started: bool) -> None:
        self._events.put((_EVENT, lambda: self._current().compacting(started)))

    def steered(self, reply_to: str) -> None:
        self._events.put((_EVENT, self._on_steered, reply_to))


def _iter_deltas(graph, message, config: dict, on_tool_use=None,
                 on_tool_result=None, message_id: str | None = None,
                 on_tool_started=None, on_tool_finished=None,
                 events: queue.Queue | None = None):
    """Yield text chunks from the graph's async stream, as a synchronous generator.

    The stream runs on the agent loop (where the MCP sessions live) and hands chunks
    over through a queue, so the caller — a plain worker thread — can drive the card
    without touching asyncio.

    `on_tool_use(tool_name) -> bool` is consulted when the model starts a tool call;
    returning True abandons the stream. Used to cut a turn short the moment the model
    reaches for a tool whose system has no grant. `on_tool_result(name, text)` receives
    each tool result, which is how the auth-wall fallback sees a server's refusal. Both run
    on the loop. `on_tool_started()` / `on_tool_finished()` run on the consumer instead, in
    stream order with the text — `events` is the queue they share, for others to join."""
    q: queue.Queue = events if events is not None else queue.Queue()

    async def _pump() -> None:
        try:
            stream = graph.astream({"messages": _turn_input(message, message_id)},
                                   config=config, stream_mode="messages")
            async for chunk, _meta in stream:
                if isinstance(chunk, ToolMessage):
                    if on_tool_result is not None:
                        on_tool_result(chunk.name, _message_text(chunk))
                    if on_tool_finished is not None:
                        q.put((_EVENT, on_tool_finished))
                    continue
                # Only the model's own words are an answer. A node's output is replayed
                # on this stream too, so the summary and any steered message — both
                # HumanMessages written by middleware — would otherwise be typed onto
                # the card; and a middleware's internal model call is not the reply.
                if not isinstance(chunk, AIMessage):
                    continue
                if (_meta or {}).get("lc_source") == "summarization":
                    continue
                # Strands emitted no tool-result event and LangGraph does; what both
                # give us is the tool call as it *starts*, which is what matters here:
                # a call to a system with no grant is certain to be refused, so seeing
                # one begin means the turn is already lost — stop rather than let the
                # model narrate the refusal at length.
                # The chunk's own words first: a tool event queued ahead of them would cut
                # the card's text block before them.
                text = _message_text(chunk)
                if text:
                    q.put(text)
                for tc in getattr(chunk, "tool_call_chunks", None) or []:
                    name = tc.get("name")
                    if name and on_tool_use is not None and on_tool_use(name):
                        return
                    if name and on_tool_started is not None:
                        q.put((_EVENT, on_tool_started))
        except Exception as e:  # noqa: BLE001 — re-raised in the consumer
            q.put(e)
        finally:
            q.put(turns._STREAM_END)

    asyncio.run_coroutine_threadsafe(_pump(), turns._LOOP)
    while True:
        item = q.get()
        if item is turns._STREAM_END:
            return
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple) and item[0] is _EVENT:
            item[1](*item[2:])
            continue
        yield item


def _stream_to_chat(session: dict, message, chat_id: str,
                    message_id: str | None = None, received_ms: int | None = None) -> str:
    """Run the turn, typing the answer into a streaming card as it is produced, and
    return the full text. If the card can't be created or an update fails (e.g. the
    cardkit:card:write scope isn't granted), fall back to posting the final text — the
    answer must survive regardless. Consent replies are handled by the caller."""
    session["tool_texts"] = []
    # One card per stretch of the turn: a message steered in mid-turn closes the current
    # card and the rest is answered on a new one, below that message.
    cards = [lark_card.StreamingCard(chat_id)]
    streaming = cards[0].open()
    acc: list[str] = []         # the current card's text
    earlier: list[str] = []     # what the closed cards said, for the return value
    group = None                # asked once, and only if a message is steered in

    def _deliver(text: str, last: bool) -> None:
        """Finish the current card with `text`, or post it as text if the card failed."""
        nonlocal streaming
        card = cards[-1]
        if streaming:
            if not card.close(text, handed_over=not last):
                streaming = False
        elif getattr(card, "sent", False):
            card.close(text)    # failed mid-stream: stops its worker, clears its live line
        if not streaming and text:
            # Either CardKit was never available or it failed mid-stream. Post the text so
            # the user still gets it.
            if not lark_card.send_text(chat_id, text):
                log.error("could not deliver reply to %s: %s", chat_id, text[:200])

    def _on_steered(reply_to: str) -> None:
        nonlocal streaming, acc, group
        # A card that has shown nothing yet is not handed over: it already sits where the
        # answer belongs, and closing it would leave an empty card behind.
        if not "".join(acc).strip() and not getattr(cards[-1], "_tools_done", 0):
            return
        try:
            _deliver("".join(acc), last=False)
            earlier.append("".join(acc))
            acc = []
            if group is None:
                group = lark_card.is_group(chat_id)
            # Quoted in a group, so others can see which message is being answered.
            cards.append(lark_card.StreamingCard(
                chat_id, reply_to=reply_to if group else "", carry=cards[-1]))
            streaming = cards[-1].open()
        except Exception:  # noqa: BLE001 — the rest of the answer must still arrive
            log.warning("could not hand the turn over to a new card", exc_info=True)
            streaming = False

    thread_id = (session.get("config") or {}).get("configurable", {}).get("thread_id", "")
    events: queue.Queue = queue.Queue()
    in_order = _InStreamOrder(events, lambda: cards[-1], _on_steered)
    if thread_id:
        turns._turn_cards[thread_id] = in_order
    try:

        # A tool whose system has no grant is certain to be refused, and letting the turn run
        # on means the model narrates that refusal at length before the consent card arrives.
        # Stop at the tool call instead — the caller sees `walled_tool` and posts the card.
        def _abort_on_ungranted_tool(tool_name: str) -> bool:
            idp = grants._walled_idp(session, tool_name)
            if not idp:
                return False
            session["walled_tool"] = tool_name
            log.info("aborting turn: %s needs %s consent", tool_name, idp)
            return True

        def _on_tool_result(name: str, text: str) -> None:
            session["tool_texts"].append((name, text))

        for delta in _iter_deltas(session["graph"], message,
                                  turns._run_config(session, delivery.LarkDelivery(chat_id),
                                                    received_ms),
                                  on_tool_use=_abort_on_ungranted_tool,
                                  on_tool_result=_on_tool_result, message_id=message_id,
                                  on_tool_started=lambda: cards[-1].tool_started(),
                                  on_tool_finished=lambda: cards[-1].tool_finished(),
                                  events=events):
            acc.append(delta)
            if streaming and not cards[-1].update("".join(acc)):
                streaming = False  # a failed write drops us to the fallback below
        text = "".join(acc)

        # Aborted for consent: whatever the model had started saying is a half-sentence
        # about a failure the user is about to be asked to fix, so replace it rather than
        # leave it on the card.
        if session.get("walled_tool"):
            idp = grants._walled_idp(session, session["walled_tool"]) or "lark"
            text = f"🔐 这一步需要访问你的 {grants._label(idp)} 账号"

        _deliver(text, last=True)
        return "".join(earlier) + text
    except BaseException:
        # The stream failed: stop the card claiming it is still working.
        if streaming:
            _quietly(cards[-1].close, "".join(acc))
        raise
    finally:
        # Only this turn's card: a later turn may already have registered its own.
        if turns._turn_cards.get(thread_id) is in_order:
            turns._turn_cards.pop(thread_id, None)
