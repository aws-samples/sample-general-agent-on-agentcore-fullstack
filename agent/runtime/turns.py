"""One turn per conversation at a time, across surfaces: claiming a thread, steering a message
into the turn already running on it, and releasing it. Also the process's one event loop,
which every MCP session is bound to.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time

from langchain_core.messages import HumanMessage

from channels import delivery
from channels import images

log = logging.getLogger("agent.turns")

# --------------------------- steering a running turn -------------------------
# A message that arrives while a turn is running used to be refused: the turn had written
# a toolUse whose toolResult was not there yet, and Bedrock rejects that whole history —
# which read as "the agent is broken" for as long as the turn lasted. Refusing was honest
# but made the user re-send. Instead the message is handed to the turn already running.
#
# Where it may be injected is not a free choice. A toolResult must follow its toolUse
# immediately, so appending a HumanMessage between them would recreate exactly the failure
# this fixes. `before_model` is the one safe boundary: any tool results from the previous
# step are written, and the next model call has not happened yet.
_pending_steer: dict[str, list[tuple[str, str, bool, int]]] = {}


# The Lark message last steered into each thread, for the card that answers it to reply to.
_steer_reply_to: dict[str, str] = {}


def steer(thread_id: str, message: str, message_id: str = "", fresh: bool = False) -> None:
    """Hand a message to the turn already running on this thread. `message_id` is the id
    its attachments were registered under (images.register), so its pixels find it.

    `fresh` marks the replay that follows a consent. It is never injected into the running
    turn — that turn's session predates the grant and would wall again — only answered as a
    continuation, on a rebuilt session."""
    with _in_flight_lock:
        _pending_steer.setdefault(thread_id, []).append(
            (message, message_id, fresh, int(time.time() * 1000)))


def _take_steer(thread_id: str) -> list[tuple]:
    with _in_flight_lock:
        _steer_reply_to.pop(thread_id, None)    # answered as a continuation, on its own card
        return _pending_steer.pop(thread_id, [])


# The card of the turn running on a thread, for what happens inside the graph that the
# token stream cannot show — a summary is generated without streaming a word. Per thread,
# which holds because a thread runs one turn at a time.
_turn_cards: dict[str, object] = {}


# Background turns in flight. AgentCore may reclaim an idle container, which would
# kill them, so /ping reports HealthyBusy while this is non-zero. Threads are tracked
# individually as well: a second turn on a thread that is mid tool call reads a checkpoint
# whose toolUse has no toolResult yet, and Bedrock rejects it — the state is *in flight*,
# not orphaned, so it must be waited out rather than repaired.
_in_flight = 0


# thread → the surface running a turn on it. The surface decides what a second message
# does: on the same surface it is steered into the running turn, whose answer appears where
# the user is looking; across surfaces it would appear in the other window, so it is told.
_in_flight_threads: dict[str, str] = {}


# Re-entrant: claiming a thread, registering a message's attachments and queueing a steer
# happen as one step under it, and steer() takes it too.
_in_flight_lock = threading.RLock()


def busy() -> bool:
    with _in_flight_lock:
        return _in_flight > 0


def _track(delta: int, thread_id: str = "", surface: str = "lark", dest: str = "") -> None:
    global _in_flight
    with _in_flight_lock:
        _in_flight += delta
        if thread_id:
            if delta > 0:
                _in_flight_threads[thread_id] = surface
                _claim_dest[thread_id] = dest
            else:
                _in_flight_threads.pop(thread_id, None)
                _claim_dest.pop(thread_id, None)


# Where the running turn's answer goes — a Lark chat, or one web page's stream. A message is
# steered into a turn only if its answer would appear where it was sent: one user's DM and
# group chats share a thread, and so do every tab and device a web user has open.
_claim_dest: dict[str, str] = {}


def thread_busy(thread_id: str) -> bool:
    """Whether a turn is already running on this thread, in this container."""
    with _in_flight_lock:
        return thread_id in _in_flight_threads


def thread_owner(thread_id: str) -> str | None:
    """The surface running a turn on this thread, or None."""
    with _in_flight_lock:
        return _in_flight_threads.get(thread_id)


def try_claim(thread_id: str, surface: str, dest: str = "") -> str | None:
    """Claim the thread for a turn on `surface`. Returns None on success, or the surface
    already holding it. One step under the lock: a separate check and mark left seconds
    between them (a cold session build), and two turns could start on one thread."""
    with _in_flight_lock:
        owner = _in_flight_threads.get(thread_id)
        if owner is None:
            _track(+1, thread_id, surface, dest)
        return owner


def claim_or_steer(thread_id: str, surface: str, dest: str, message: str,
                   message_id: str = "") -> str | None:
    """Claim the thread, or hand `message` to the turn already answering into `dest` — one
    step under the lock, as chat_async does for Lark, so the message cannot land in a turn
    that has just released. None when claimed, "steered" when handed over, f"{surface}-other"
    when the turn answers into another of this surface's destinations, else the other
    surface that holds the thread."""
    with _in_flight_lock:
        owner = _in_flight_threads.get(thread_id)
        if owner is None:
            _track(+1, thread_id, surface, dest)
            return None
        if owner != surface:
            return owner
        if not dest or _claim_dest.get(thread_id) != dest:
            return f"{surface}-other"
        steer(thread_id, message, message_id)
        return "steered"


def take_steer_or_release(thread_id: str) -> list[tuple]:
    """What was steered in after the turn's last model call, to answer as a continuation —
    or, when nothing was, the release, in the same step: a message arriving in between
    would otherwise be steered into a turn that no longer reads it."""
    with _in_flight_lock:
        queued = _take_steer(thread_id)
        if not queued:
            release(thread_id)
        return queued


def release(thread_id: str) -> list[tuple[str, str, bool]]:
    """End a claim: forget the turn's attachments, then free the thread — in that order,
    so a message arriving right after cannot register into a turn that has finished.

    Returns whatever was still queued to be steered in. Dropped here rather than left for
    the next turn, which would inject it after its own message with its pixels gone and
    its attachment numbers pointing at that turn's files. A no-op if the thread is not
    claimed, so a second release cannot drive the in-flight count below zero."""
    with _in_flight_lock:
        if thread_id not in _in_flight_threads:
            return []
        leftover = _pending_steer.pop(thread_id, [])
        _steer_reply_to.pop(thread_id, None)
        images.end_turn(thread_id)
        _track(-1, thread_id)
        return leftover


_BUSY_ELSEWHERE = {"web": "这段对话正在网页上回答，等它结束后再发送。",
                   "lark": "这段对话正在 Lark 里回答，等它结束后再发送。",
                   "web-other": "这段对话正在另一个窗口里回答，等它结束后再发送。",
                   "lark-other": "这段对话正在另一个聊天里回答，等它结束后再发送。",
                   "sync": "这段对话正在另一处回答，等它结束后再发送。"}


# --------------------------- the process's one event loop --------------------
# MCP sessions are anyio-based and bound to the loop that opened them, and we keep a
# session open across turns — so every await touching one has to happen on that same
# loop. One long-lived loop in a daemon thread gives us that, and keeps this module's
# API synchronous for the aiohttp handlers (which call it in an executor), the card
# writer thread, and the tests. A loop per turn, which the previous Strands
# implementation could get away with, would strand the cached sessions.

_LOOP = asyncio.new_event_loop()


threading.Thread(target=_LOOP.run_forever, name="agent-loop", daemon=True).start()


def _run(coro, timeout: float | None = None):
    """Run a coroutine on the agent loop from a synchronous caller."""
    return asyncio.run_coroutine_threadsafe(coro, _LOOP).result(timeout)


_STREAM_END = object()


def _run_config(session: dict, channel: delivery.Delivery, received_ms: int | None = None,
                account: str = "") -> dict:
    """The session's config plus where this run may deliver a file.

    A fresh dict every run: the session is cached per actor and thread and is served to
    every surface, so a target stored on it would send one surface's file to another's.
    A turn's attachments are not here: a message steered in mid-turn joins a run whose
    config is already fixed, so they live in images' per-thread turn registry instead.
    """
    configurable = session["config"]["configurable"]
    return {**session["config"], "recursion_limit": _STEP_LIMIT,
            "configurable": {**configurable, "delivery": channel,
                             "received_ms": received_ms or int(time.time() * 1000),
                             # What the person signed in as, from their verified token.
                             "account": account}}


def _received_ms() -> int | None:
    """When the running turn arrived, from its run config; None outside a run."""
    try:
        from langgraph.config import get_config
        return ((get_config() or {}).get("configurable") or {}).get("received_ms")
    except Exception:  # noqa: BLE001 — outside a graph run there is no turn
        return None


# Explicit, because the AG-UI adapter fills an unset limit with langchain_core's 25 — about a
# dozen tool calls — while LangGraph's own default, which a Lark turn gets, is 10007.
_STEP_LIMIT = int(os.environ.get("TURN_STEP_LIMIT", "10007"))


def _said(item: tuple) -> HumanMessage:
    """A queued (text, id, fresh[, arrived ms]) entry as the message it becomes, carrying
    when it arrived: a continuation answers it later than that."""
    ts = item[3] if len(item) > 3 else None
    return HumanMessage(item[0], id=item[1] or None,
                        additional_kwargs={"ts": ts} if ts else {})


async def arun_in_thread(fn, *args):
    """Run a blocking call off the caller's event loop."""
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


def stream_on_agent_loop(agen_factory, on_done=None, until=None):
    """Start an async generator on _LOOP now, and return an async iterator over its items
    for another loop to consume.

    The graph's MCP sessions are bound to the loop that opened them, so anything awaiting
    them has to run on _LOOP — but an aiohttp handler runs on the server's own loop; the
    hand-off is a thread-safe queue. Started eagerly, and `on_done` runs on _LOOP when the
    producer finishes, not when the consumer stops: a web page that is closed mid-answer
    stops reading, but the graph runs on and keeps writing the checkpoint, so anything
    marking the turn as running has to last until the producer is done. An item `until`
    accepts ends the run there: it is not passed on, and the generator is closed. Items
    `on_done` returns are passed on last."""
    q: queue.Queue = queue.Queue()

    async def _pump() -> None:
        try:
            agen = agen_factory()
            async for item in agen:
                if until is not None and until(item):
                    await agen.aclose()
                    break
                q.put(item)
        except Exception as e:  # noqa: BLE001 — re-raised in the consumer
            q.put(e)
        finally:
            # Before the end of the stream, so whatever on_done hands back reaches the
            # consumer: the release reports messages it had to drop.
            if on_done is not None:
                try:
                    for item in on_done() or []:
                        q.put(item)
                except Exception:  # noqa: BLE001 — never lose the release to a logging bug
                    log.exception("on_done failed")
            q.put(_STREAM_END)

    asyncio.run_coroutine_threadsafe(_pump(), _LOOP)

    async def _drain():
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _STREAM_END:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    return _drain()
