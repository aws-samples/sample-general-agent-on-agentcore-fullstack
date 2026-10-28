"""What runs around each model call: messages steered in mid-turn, pictures attached to the
request only, when each message was said, and the summarisation that bounds an endless
thread. core._middleware() decides which of these the agent runs.
"""

from __future__ import annotations

import logging
import os
import time

from langchain_core.messages import AIMessage, HumanMessage

from channels import images
from runtime import telemetry
from runtime import sessions
from runtime import turns

log = logging.getLogger("agent.middleware")

# --------------------------- bounding an endless thread ----------------------
# The thread is permanent per user and DynamoDBSaver has no prune, so summarisation is the
# only bound. Threshold-triggered, never per turn: it rewrites the cached prefix.
# Tokens, not a fraction: a fraction of a permanently growing thread re-summarises
# forever. Late, because summarising early discards detail the user can still refer to.
_SUMMARIZE_AT_TOKENS = int(os.environ.get("SUMMARIZE_AT_TOKENS", "120000"))  # 0 disables


_SUMMARIZE_KEEP = int(os.environ.get("SUMMARIZE_KEEP_MESSAGES", "20"))


# What langchain's SummarizationMiddleware writes in place of a summary when it could not
# make one; applying it would replace the whole history with this line.
_SUMMARY_FALLBACK = "Previous conversation was too long to summarize."


# A summary of ~100k tokens of history is one long non-cached call; botocore's 60 s read
# timeout would cut it off.
_SUMMARY_TIMEOUT = int(os.environ.get("SUMMARY_TIMEOUT_SECONDS", "300"))


def _is_summary(message) -> bool:
    """The message SummarizationMiddleware puts at the head of the thread."""
    return (isinstance(message, HumanMessage)
            and (message.additional_kwargs or {}).get("lc_source") == "summarization")


def _summary_is_usable(update) -> bool:
    """False when the update would replace the history with the fallback line."""
    for m in (update or {}).get("messages", []):
        if _is_summary(m):
            return not sessions._text_of(m).rstrip().endswith(_SUMMARY_FALLBACK)
    return True


def _thread_of(runtime) -> str:  # noqa: ANN001 — the framework's own type
    """This run's thread id. `runtime.config` does not exist on LangGraph 1.2, so reading
    it silently disabled steering; `execution_info` is where the id actually lives."""
    return getattr(getattr(runtime, "execution_info", None), "thread_id", "") or ""


# Bedrock's per-request image limit is about 20; a turn that has had pictures steered into
# it can gather more than that.
_MAX_IMAGES_PER_REQUEST = 20


class _ImageMiddleware:
    """Puts the pixels into the model call without putting them into the conversation.

    An image costs tokens by area on every request that carries it, and the thread is
    replayed each turn, so an inlined screenshot would be billed for the life of the
    conversation. The stored message keeps a handle; the bytes are fetched for the turn
    the image arrived in and attached here, to the request only. Nothing to evict later,
    so the cache prefix is never rewritten."""

    name = "images"

    def _attach(self, request):  # noqa: ANN001 — the framework's own type
        pixels = images.pixels_for(_thread_of(request.runtime))
        if not pixels or not request.messages:
            return request
        out, room = [], _MAX_IMAGES_PER_REQUEST
        # Newest first, so if a turn has gathered more pictures than one request may carry,
        # the ones dropped are the oldest.
        for m in reversed(request.messages):
            blocks = pixels.get(getattr(m, "id", None)) if isinstance(m, HumanMessage) else None
            blocks = (blocks or [])[:max(room, 0)]
            room -= len(blocks)
            if blocks:
                text = m.content if isinstance(m.content, str) else ""
                # Same id, so this replaces the message in the request rather than adding
                # to the thread. The stored copy keeps its placeholders.
                m = HumanMessage(id=m.id, content=(
                    [{"type": "text", "text": text}] if text else []) + list(blocks))
            out.append(m)
        request.messages = out[::-1]
        return request

    def wrap_model_call(self, request, handler):  # noqa: ANN001 — the framework's signature
        return handler(self._attach(request))

    async def awrap_model_call(self, request, handler):  # noqa: ANN001 — same
        return await handler(self._attach(request))


class _ClockMiddleware:
    """Records when each message of the thread was said, for a page restoring the history.

    In `additional_kwargs`, which Converse never sends to the model. Stamped around the model
    call rather than in a hook of its own: a hook is a graph node, and every node is one more
    checkpoint write per model call."""

    name = "clock"

    @staticmethod
    def _stamped(request, started_ms: int, response):  # noqa: ANN001, ANN205 — framework types
        from langchain.agents.middleware.types import ExtendedModelResponse, ModelResponse
        from langgraph.types import Command
        result = response.result if isinstance(response, ModelResponse) else [response]
        for m in result:
            if isinstance(m, AIMessage):
                m.additional_kwargs.setdefault("ts", started_ms)
        # What the user said since the last reply, from the state: request.messages may hold
        # the image middleware's copies, which carry the pixels the thread must not keep.
        # Messages built here carry their arrival already (_said); what is left is a turn's
        # own message as a surface handed it in — AG-UI's — which arrived with the turn.
        at = turns._received_ms() or started_ms
        said = []
        for m in reversed((getattr(request, "state", None) or {}).get("messages") or []):
            if isinstance(m, AIMessage):
                break
            if isinstance(m, HumanMessage) and m.id and "ts" not in m.additional_kwargs \
                    and not _is_summary(m):
                said.append(m.model_copy(update={
                    "additional_kwargs": {**m.additional_kwargs, "ts": at}}))
        if not said:
            return response
        # Same ids, so the reducer replaces them in place.
        return ExtendedModelResponse(
            model_response=response if isinstance(response, ModelResponse)
            else ModelResponse(result=result),
            command=Command(update={"messages": said[::-1]}))

    def wrap_model_call(self, request, handler):  # noqa: ANN001 — the framework's signature
        started = int(time.time() * 1000)
        return self._stamped(request, started, handler(request))

    async def awrap_model_call(self, request, handler):  # noqa: ANN001 — same
        started = int(time.time() * 1000)
        return self._stamped(request, started, await handler(request))


class _SteeringMiddleware:
    """Injects messages that arrived mid-turn, at the only boundary where it is legal."""

    name = "steering"

    def before_model(self, state, runtime):  # noqa: ANN001 — the framework's signature
        thread_id = _thread_of(runtime)
        if not thread_id:
            return None
        # A consent replay stays queued for the continuation; everything else joins now.
        with turns._in_flight_lock:
            queued = turns._pending_steer.get(thread_id) or []
            now = [q for q in queued if not q[2]]
            if not now:
                return None
            later = [q for q in queued if q[2]]
            if later:
                turns._pending_steer[thread_id] = later
            else:
                turns._pending_steer.pop(thread_id, None)
            reply_to = turns._steer_reply_to.pop(thread_id, "")
        log.info("steering %d message(s) into the running turn on %s", len(now), thread_id)
        telemetry.count("steer_injected")
        # What follows answers the new message, so it goes below it, on a card of its own.
        try:
            card = turns._turn_cards.get(thread_id)
            if card is not None:
                card.steered(reply_to)
        except Exception:  # noqa: BLE001 — the card is decoration, the message is not
            log.warning("could not hand the card over", exc_info=True)
        return {"messages": [turns._said(q) for q in now]}


def _turn_status(kind: str, started: bool) -> None:
    """Tell the running turn's card what the graph is doing. Best-effort and silent outside
    a turn: /compact summarises with no run config and no card.

    Everything is inside the try, attribute lookups included: this runs inside the
    summariser, and an exception escaping here would be taken for a failed summary and
    skip the compaction — a status line costing the thing it reports on."""
    try:
        from langgraph.config import get_config
        thread_id = ((get_config() or {}).get("configurable") or {}).get("thread_id", "")
    except Exception:  # noqa: BLE001 — outside a graph run, as /compact is: nothing to tell
        return
    try:
        card = turns._turn_cards.get(thread_id)
        if card is not None and kind == "compacting":
            card.compacting(started)
    except Exception:  # noqa: BLE001 — never let the status line cost the summary
        log.warning("could not report %s to the card", kind, exc_info=True)


def _summarization(model, trigger=None, keep=None):
    """SummarizationMiddleware that never discards history it failed to summarise.

    `trim_tokens_to_summarize=None`: the library default trims what it summarises to the
    last 4,000 tokens starting at a human message, and an agentic tail — one request
    followed by many tool results — has none in that window, so the trim comes back empty
    and the fallback line replaces the whole history (measured on a real thread: 114
    messages lost). The full slice fits the model's window; on overflow the middleware
    retries with the middle cut out.

    A failed or fallback summary skips this turn's compaction instead of applying it: the
    thread is still within the model's window, and the next turn tries again. Named like
    the base class so the graph node and checkpoints keep their names."""
    from langchain.agents.middleware import SummarizationMiddleware as _Base

    def _kept(update):
        if _summary_is_usable(update):
            return update
        log.warning("summarisation produced no summary; keeping the history as is")
        telemetry.count("summarize_failed", reason="fallback")
        return None

    class SummarizationMiddleware(_Base):
        # Only these two run when a summary is really being written, so the card says so
        # for exactly that stretch and not for every threshold check.
        def _create_summary(self, messages):  # noqa: ANN001 — the framework's signature
            _turn_status("compacting", True)
            try:
                return super()._create_summary(messages)
            finally:
                _turn_status("compacting", False)

        async def _acreate_summary(self, messages):  # noqa: ANN001 — same
            _turn_status("compacting", True)
            try:
                return await super()._acreate_summary(messages)
            finally:
                _turn_status("compacting", False)

        def before_model(self, state, runtime):  # noqa: ANN001 — the framework's signature
            try:
                return _kept(super().before_model(state, runtime))
            except Exception as e:  # noqa: BLE001 — a failed summary must not fail the turn
                log.exception("summarisation failed; keeping the history as is")
                telemetry.count("summarize_failed", reason=type(e).__name__)
                return None

        async def abefore_model(self, state, runtime):  # noqa: ANN001 — same
            try:
                return _kept(await super().abefore_model(state, runtime))
            except Exception as e:  # noqa: BLE001 — same
                log.exception("summarisation failed; keeping the history as is")
                telemetry.count("summarize_failed", reason=type(e).__name__)
                return None

    return SummarizationMiddleware(
        model=model,
        trigger=trigger or ("tokens", _SUMMARIZE_AT_TOKENS),
        keep=keep or ("messages", _SUMMARIZE_KEEP),
        trim_tokens_to_summarize=None,
    )
