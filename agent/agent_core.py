"""Core agent: a LangGraph ReAct agent on Bedrock, with AgentCore Memory for session
continuity and per-user MCP tool identity.

Server-reuse model: the model, the MCP sessions and the compiled graph are built ONCE
per session and cached — not rebuilt per message. Rebuilding per message re-handshakes
every MCP server and re-lists tools, which adds ~15–20s of latency. AgentCore gives
each session its own microVM, so the cache holds essentially one entry per container.

Memory is two independent backends, because the two jobs have different requirements
(see docs/architecture.md, "Conversation memory"):
  - DynamoDBSaver — the checkpointer, i.e. the conversation itself. Full graph state
    (messages with their tool calls and results, channels, pending writes), addressed by
    thread_id alone — the sha256 of the actor. Nothing about the container is part of the
    key, so a turn cut off at the 15-minute invoke cap resumes in a brand-new microVM.
    Written per superstep, so the durable record advances *during* a turn: there is no
    shutdown hook on a microVM, and a design that flushes at the end loses the turn.
  - AgentCore Memory — long-term records only, reached through the `remember`/`recall`
    tools in memory_tools. Nothing is written per turn: retrieval is the priced operation,
    so storing every exchange and querying blindly is the pattern that makes cost scale
    with chat volume instead of with need. What to keep is the model's explicit decision.

Per-user Lark access (agent-side 3LO): for each end-user the agent fetches that user's
vaulted Lark token from AgentCore Identity (GetResourceOauth2Token, USER_FEDERATION)
and connects directly to the lark-mcp Runtime, passing the token in a custom header
(the server reads it and calls Lark as that user). The token gates tool *calls*, not
the connection — lark-mcp lists its tools without one — so an unauthorized user gets a
working session and is only asked to consent if the model actually reaches for a Lark
tool. 3LO is agent-side because a Runtime-hosted MCP server cannot be handed a per-user
token by the Gateway at all (see docs/agentcore-behavior.md).

`chat_result` returns the final text; `chat_async` streams it into a Lark card.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import queue
import threading
import time
import uuid
from contextlib import AsyncExitStack

from botocore.config import Config

from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
# At module level: the tool's `config: RunnableConfig` is resolved here for injection.
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.sessions import create_session
from langchain_mcp_adapters.tools import load_mcp_tools

import code_tools
import deliveries
import delivery
import images
import lark_3lo
import lark_notify
import memory_tools
import telemetry
import websearch

log = logging.getLogger("agent.core")

_REGION = os.environ.get("AWS_REGION", "us-west-2")
_MODEL_ID = os.environ["BEDROCK_MODEL_ID"]  # injected at deploy, from MODEL_ID in .env
_MEMORY_ID = os.environ.get("BEDROCK_AGENTCORE_MEMORY_ID", "")
_CHECKPOINT_TABLE = os.environ.get("CHECKPOINT_TABLE", "")
_CHECKPOINT_BUCKET = os.environ.get("CHECKPOINT_BUCKET", "")
# One thread per user, rotated only by /reset, so the state has to expire on its own.
_CHECKPOINT_TTL = int(os.environ.get("CHECKPOINT_TTL_DAYS", "365")) * 86400
# Empty unless ./deploy.sh mcp approval ran — the approval tools are opt-in.
_APPROVAL_MCP_URL = os.environ.get("APPROVAL_MCP_URL", "")
# The second downstream system. Empty unless Google is configured; the agent then simply has
# no Google tools, the same way it has no search without a gateway.
_GOOGLE_MCP_URL = os.environ.get("GOOGLE_MCP_URL", "")
_GOOGLE_PROVIDER = os.environ.get("GOOGLE_OAUTH_PROVIDER", "")
_GOOGLE_SCOPES = os.environ.get(
    "GOOGLE_SCOPES", "openid email profile https://www.googleapis.com/auth/calendar.readonly").split()
# Systems besides Lark, keyed like the router's IDP_REGISTRY → (MCP url, provider, scopes).
# Only configured ones appear, so the rest of the code never asks "is Google set up?".
_DOWNSTREAMS = {k: v for k, v in {
    "google": (_GOOGLE_MCP_URL, _GOOGLE_PROVIDER, _GOOGLE_SCOPES),
}.items() if v[0] and v[1]}
# Short on purpose. Operational facts a model cannot infer — the workspace persists, whether
# there is internet — live in each tool's description, next to what they constrain, instead of
# being paid for on every turn. Debugging heuristics do not belong here at all: they encode
# one deployment's incident history into a general-purpose agent.
_SYSTEM = os.environ.get(
    "AGENT_SYSTEM_PROMPT",
    "You are a general-purpose agent. Be concise. Use the tools when they help. "
    "Report what a tool actually returned, and never present work you did not "
    "complete as done.",
)
# Rebuild a cached session before its Cognito access token (~1h) expires.
_SESSION_TTL = int(os.environ.get("SESSION_TTL_SECONDS", "3000"))  # 50 min
# An unauthorized session is cached only briefly: it works, but the moment the user
# consents we want the next turn to pick up the token.
_UNAUTH_TTL = int(os.environ.get("UNAUTH_SESSION_TTL_SECONDS", "60"))

# Prompt caching on Bedrock is explicit or implicit by model, not by choice. On Converse only
# Claude and Nova take a cachePoint, and they cache nothing without one; every other model
# caches matching prefixes on its own or not at all, and may reject a request that carries a
# cachePoint (Kimi K3 does, measured). So the model decides whether to send one, and
# PROMPT_CACHE_TTL only sets how long: "5m" (Bedrock's default) | "1h" | "off".
# Source: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
# Placement, TTL choice and the measured break-even: docs/architecture.md.
_EXPLICIT_CACHE_FAMILIES = ("anthropic.claude", "amazon.nova")


def _caches_explicitly(model_id: str) -> bool:
    """Whether this model caches only behind a cachePoint (and accepts one)."""
    return any(f in model_id.lower() for f in _EXPLICIT_CACHE_FAMILIES)


_CACHE_SETTING = os.environ.get("PROMPT_CACHE_TTL", "").strip() or "5m"
if _CACHE_SETTING not in ("5m", "1h", "off"):
    raise SystemExit(f"PROMPT_CACHE_TTL must be 5m, 1h or off, not {_CACHE_SETTING!r}")
# The TTL actually put on requests: empty sends no cachePoint at all.
_CACHE_TTL = (_CACHE_SETTING if _caches_explicitly(_MODEL_ID) and _CACHE_SETTING != "off"
              else "")


class _CachingChatBedrockConverse(ChatBedrockConverse):
    """ChatBedrockConverse that asks for prompt caching on every request.

    langchain-aws does the placement AWS documents — after the system prompt, after the
    tool definitions, and a rolling pair at the end of the message list, up to Bedrock's
    limit of four. All we supply is the intent.

    Before tuning this, know that the 1,024-token minimum for Sonnet 5 is cumulative over
    tools + system + messages in that order. Measured against the deployed servers, our
    tool definitions are ~1,618 tokens with the approval server and ~400 without it, and
    the system prompt is ~32 — so on a minimal deployment the tools checkpoint earns
    nothing until the conversation itself grows past the minimum.
    """

    cache_ttl: str

    def _with_cache(self, kwargs: dict) -> dict:
        # An explicit per-call value wins: a caller asking for something specific should
        # not be silently overridden.
        if self.cache_ttl and "cache_control" not in kwargs:
            kwargs["cache_control"] = {"ttl": self.cache_ttl}
        return kwargs

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return super()._generate(messages, stop=stop, run_manager=run_manager,
                                 **self._with_cache(kwargs))

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        return super()._stream(messages, stop=stop, run_manager=run_manager,
                               **self._with_cache(kwargs))


# The class defines no _agenerate/_astream, so the async paths run these in a thread —
# overriding the two synchronous entry points covers streaming and non-streaming, sync
# and async alike.
# A model may send a tool call's arguments in one piece once they are all generated, so the
# stream is silent while it writes a large file: Kimi K3 sent 13 KB after 18.5 s of silence
# (measured), and botocore's 60 s default cut a longer one off. Connecting keeps the default.
_MODEL_READ_TIMEOUT = 300
_model = _CachingChatBedrockConverse(model=_MODEL_ID, region_name=_REGION,
                                     cache_ttl=_CACHE_TTL,
                                     config=Config(read_timeout=_MODEL_READ_TIMEOUT))

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
            return not _text_of(m).rstrip().endswith(_SUMMARY_FALLBACK)
    return True


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
        at = _received_ms() or started_ms
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
        with _in_flight_lock:
            queued = _pending_steer.get(thread_id) or []
            now = [q for q in queued if not q[2]]
            if not now:
                return None
            later = [q for q in queued if q[2]]
            if later:
                _pending_steer[thread_id] = later
            else:
                _pending_steer.pop(thread_id, None)
            reply_to = _steer_reply_to.pop(thread_id, "")
        log.info("steering %d message(s) into the running turn on %s", len(now), thread_id)
        telemetry.count("steer_injected")
        # What follows answers the new message, so it goes below it, on a card of its own.
        try:
            card = _turn_cards.get(thread_id)
            if card is not None:
                card.steered(reply_to)
        except Exception:  # noqa: BLE001 — the card is decoration, the message is not
            log.warning("could not hand the card over", exc_info=True)
        return {"messages": [_said(q) for q in now]}


def _middleware() -> list:
    from langchain.agents.middleware import AgentMiddleware
    # Subclassed here rather than at import time so the class exists only when the
    # framework does, matching how summarisation is imported.
    steering = type("SteeringMiddleware", (AgentMiddleware,),
                    dict(_SteeringMiddleware.__dict__))()
    # First in the list: the image bytes are attached to the outgoing request only, so
    # anything after this — summarisation especially — works on the placeholders.
    pictures = type("ImageMiddleware", (AgentMiddleware,),
                    dict(_ImageMiddleware.__dict__))()
    clock = type("ClockMiddleware", (AgentMiddleware,), dict(_ClockMiddleware.__dict__))()
    if not _SUMMARIZE_AT_TOKENS:
        return [pictures, steering, clock]
    return [pictures, steering, clock, _summarization(_summary_model())]


def _summary_model():
    """The model that writes summaries, for the threshold and for /compact alike."""
    return _CachingChatBedrockConverse(
        model=_MODEL_ID, region_name=_REGION,
        # Caching deliberately off for this one: the summary call happens once per
        # threshold crossing and is never re-read, so a cachePoint would only buy a
        # 1.25-2x write premium.
        cache_ttl="",
        # The summary is generated inside the graph, so without these its tokens stream
        # into the chat like an answer: `nostream` keeps it out of LangGraph's messages
        # stream (the Lark card), `emit-messages` out of AG-UI (the web page).
        tags=["nostream"], metadata={"emit-messages": False},
        timeout=_SUMMARY_TIMEOUT)


# The card of the turn running on a thread, for what happens inside the graph that the
# token stream cannot show — a summary is generated without streaming a word. Per thread,
# which holds because a thread runs one turn at a time.
_turn_cards: dict[str, object] = {}


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
        card = _turn_cards.get(thread_id)
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


# session_id -> {graph, config, stack, created, ...}. One microVM ≈ one session.
_sessions: dict[str, dict] = {}
_lock = threading.Lock()

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


def model_id() -> str:
    """What this container actually calls Bedrock with — /status reads it from here so a
    stale router config cannot report a model nobody is using."""
    return _MODEL_ID


def cache_mode() -> str:
    """How this model's prompt cache works here: "explicit 5m" / "explicit 1h" (a cachePoint
    on every request), "implicit" (the model caches by itself) or "off"."""
    if _CACHE_TTL:
        return f"explicit {_CACHE_TTL}"
    return "off" if _caches_explicitly(_MODEL_ID) else "implicit"


def model_settings() -> dict:
    """What shapes a turn in this container, as it is actually configured. The prompt
    itself is not returned: it can be long, and its length says whether it was changed."""
    return {"model": _MODEL_ID, "region": _REGION, "cache": cache_mode(),
            "summarizeAt": _SUMMARIZE_AT_TOKENS, "summarizeKeep": _SUMMARIZE_KEEP,
            "systemPromptCustom": "AGENT_SYSTEM_PROMPT" in os.environ
                                  and bool(os.environ["AGENT_SYSTEM_PROMPT"].strip()),
            "systemPromptChars": len(_SYSTEM)}


def tools_overview(session: dict) -> dict:
    """The tools this session really has, in the order they were assembled, each with
    where it came from and which system's grant it needs ("" for none)."""
    if session.get("identity_error"):
        return {"tools": [], "identityError": session["identity_error"]}
    source, idp = session.get("tool_source") or {}, session.get("tool_idp") or {}
    return {"tools": [{"name": n, "source": s, "idp": idp.get(n, "")}
                      for n, s in source.items()],
            # A system named here has no grant yet; its tools are listed regardless,
            # because servers gate calls, not discovery.
            "ungranted": sorted((session.get("consent") or {}).keys())}


# /compact keeps fewer messages than the threshold does: someone asking for it wants the
# thread smaller now, and the threshold's keep is sized for a thread that is already huge.
_COMPACT_KEEP = int(os.environ.get("COMPACT_KEEP_MESSAGES", "6"))


def compact(actor_id: str, mem_sid: str, workload_token: str = "") -> dict:
    """Summarise the thread now, through the same middleware the threshold uses — so a
    summary that comes back as the fallback line is refused here as it is there, rather
    than replacing the history with nothing.

    Written back with update_state and no node named: measured on a create_agent graph,
    that leaves no pending step behind, and the next turn starts from the summary."""
    thread_id = mem_sid or thread_for(actor_id)
    if thread_busy(thread_id):
        return {"compacted": 0, "reason": "busy"}
    session = _get_session(actor_id, "", thread_id, workload_token=workload_token)
    if session.get("identity_error"):
        return {"compacted": 0, "reason": "identity"}
    return _run(_acompact(session["graph"], session["config"]), timeout=_SUMMARY_TIMEOUT + 30)


async def _acompact(graph, config) -> dict:
    messages = (await graph.aget_state(config)).values.get("messages") or []
    middleware = _summarization(_summary_model(), trigger=("messages", 1),
                                keep=("messages", _COMPACT_KEEP))
    # Asked first so "nothing to compact" is not reported as a failed summary. The
    # cutoff is what the middleware itself would use; langchain is pinned <2 for it.
    if middleware._determine_cutoff_index(messages) <= 0:
        return {"compacted": 0, "reason": "short", "messages": len(messages)}
    update = await middleware.abefore_model({"messages": messages}, None)
    if not update:
        return {"compacted": 0, "reason": "failed"}
    await graph.aupdate_state(config, update)
    after = (await graph.aget_state(config)).values.get("messages") or []
    kept = sum(1 for m in after if not _is_summary(m))
    telemetry.count("compacted")
    return {"compacted": len(messages) - kept, "kept": kept}


async def atools_overview(actor_id: str, mem_sid: str, workload_token: str = "") -> dict:
    """tools_overview for the session chat would use — the same cache key, so asking does
    not assemble a second session and handshake every server again."""
    session = await arun_in_thread(_get_session, actor_id, "", mem_sid or thread_for(actor_id),
                                   False, workload_token)
    return tools_overview(session)


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


def _release_unanswered(thread_id: str, chat_id: str) -> None:
    """Release after a turn that never ran, telling the chat about anything steered into it
    meanwhile — those messages were acknowledged, so dropping them silently would lose them."""
    leftover = release(thread_id)
    if leftover:
        telemetry.count("steer_unanswered", n=str(len(leftover)))
        _quietly(lark_notify.send_text, chat_id,
                 f"还有 {len(leftover)} 条消息没来得及处理，请重新发送。")


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


# actor -> the thread id the router last named for them. The router owns rotation (/reset,
# /new), and only it is trusted to say which thread an actor is on; a surface that cannot
# name one — the browser, where a claimed id would address another user's conversation —
# reads it from here instead. Same container, because both share the router's runtime
# session id, so this is in-process rather than shared state.
_thread_named: dict[str, str] = {}


def remember_thread(actor_id: str, mem_sid: str) -> None:
    if mem_sid and thread_belongs(actor_id, mem_sid):
        _thread_named[actor_id] = mem_sid


def thread_for(actor_id: str) -> str:
    """The thread this actor is on: what the router last named, else the derived default."""
    return _thread_named.get(actor_id) or _session_id_for(actor_id)


def _session_id_for(actor_id: str) -> str:
    """Deterministic per-user session id: one long conversation thread per user,
    shared across reconnects and entrypoints (STM retains it 30 days)."""
    return "sess-" + hashlib.sha256(actor_id.encode()).hexdigest()[:32]


def thread_belongs(actor_id: str, thread_id: str) -> bool:
    """Whether `thread_id` is one of this actor's threads: the derived default, or the
    default plus the 12-hex suffix the router gives a rotated one. The checkpoint is keyed
    by thread id alone, and the default is computable by anyone, so an id the caller names
    is only honoured under this prefix. Kept in step with lambda/router/identity.py."""
    default = _session_id_for(actor_id)
    if thread_id == default:
        return True
    suffix = thread_id[len(default) + 1:] if thread_id.startswith(default + "-") else ""
    return len(suffix) == 12 and all(c in "0123456789abcdef" for c in suffix)


# Shared only by the fallback: an InMemorySaver rebuilt with every session would drop
# history every time the cache expires (50 min), which is worse than the degradation the
# fallback is there to provide.
_fallback_saver = None


def _checkpointer():
    """The conversation's durable state, in DynamoDB.

    Not AgentCoreMemorySaver: that one carries a mandatory 3-365 day event expiry, bills
    per write, and needs the Memory resource to exist at all. Fidelity is identical —
    both go through JsonPlusSerializer and produce the same CheckpointTuple — so the
    choice is only about the dependency surface. AgentCore Memory is still used, for
    long-term records; see _long_term_tools().

    Built per session rather than once, because it holds a boto3 client and a stale one
    outliving its credentials is the failure that avoids."""
    global _fallback_saver
    if not _CHECKPOINT_TABLE:
        # Tests, and any runtime that predates provision writing the env var. A session
        # still works; history just dies with the container.
        if _fallback_saver is None:
            from langgraph.checkpoint.memory import InMemorySaver
            log.warning("CHECKPOINT_TABLE unset: history will not outlive this container")
            _fallback_saver = InMemorySaver()
        return _fallback_saver
    from langgraph_checkpoint_aws import DynamoDBSaver
    return DynamoDBSaver(
        table_name=_CHECKPOINT_TABLE,
        region_name=_REGION,
        ttl_seconds=_CHECKPOINT_TTL,
        # Compress first so spilling to S3 stays the exception: DynamoDB caps an item at
        # 400 KB, and one unbounded thread per user would otherwise reach it.
        enable_checkpoint_compression=True,
        s3_offload_config=(
            {"bucket_name": _CHECKPOINT_BUCKET, "key_prefix": "state"}
            if _CHECKPOINT_BUCKET else None),
    )


def _long_term_tools(actor_id: str) -> list:
    """Long-term memory reaches the model as two tools (remember/recall), not as a Store.

    A LangGraph Store would write every exchange as conversational events — the
    by-product pattern ADR 0008 rejected, and the one that makes retrieval cost scale with
    chat volume rather than with need. Nothing reads a Store here now that /status counts
    checkpoints instead of events."""
    return memory_tools.tools_for(actor_id)


_INTERRUPTED_TOOL = "The previous attempt was interrupted before this tool returned."


async def _arepair_interrupted_turn(graph, config: dict, own_claim: bool = False) -> int:
    """Answer any tool call the previous turn was killed before completing.

    Checkpoints are written per superstep, so a turn cut off between "the model emitted
    tool_calls" and "the tool returned" leaves the trailing AIMessage with calls that have
    no matching ToolMessage. Bedrock then rejects every subsequent request — each toolUse
    must have a toolResult — which is the worst failure available here: the history is
    intact but permanently unreachable, and it looks like the agent broke rather than like
    a turn was interrupted. LangGraph could resume the pending task itself, but only via
    ainvoke(None), and we always arrive carrying a new user message.

    Two shapes, two fixes. When the dangling AIMessage is still last, appending the missing
    results is right and keeps the turn's context. Once the user has sent anything else it
    is no longer last — LangGraph writes the new HumanMessage before calling the model, so a
    failed turn buries the dangling call one position deeper every time — and appending
    cannot help, because a toolResult must come *immediately* after its toolUse. There the
    only valid repair is to drop that AIMessage: its tools never ran, so it holds nothing but
    a preamble, and removing it makes the history legal again."""
    state = await graph.aget_state(config)
    messages = (getattr(state, "values", None) or {}).get("messages") or []
    dangling, answered = None, {getattr(m, "tool_call_id", None) for m in messages
                                if isinstance(m, ToolMessage)}
    for m in messages:
        if isinstance(m, AIMessage) and any(
                tc.get("id") and tc["id"] not in answered
                for tc in (getattr(m, "tool_calls", None) or [])):
            dangling = m
    if dangling is None:
        return 0
    if dangling is not messages[-1]:
        # A turn still running on this thread has exactly this shape — its toolUse is
        # written, its toolResult is not yet — so removing the message would corrupt a live
        # turn instead of repairing a dead one. Wait it out; the results are coming.
        # `own_claim`: the caller holds this thread's claim and no graph of its own is
        # running, so "busy" is itself — a claim is taken before the session is built,
        # and skipping here would leave the thread failing every turn, unrepaired.
        if thread_busy(config["configurable"]["thread_id"]) and not own_claim:
            log.info("dangling call belongs to a turn still in flight; leaving it alone")
            return 0
        # Buried and nothing is running: drop it rather than paper over it in the wrong
        # position. Appending cannot help, a toolResult must follow its toolUse directly.
        from langchain_core.messages import RemoveMessage
        log.warning("dropping a buried AIMessage with %d unanswered tool call(s)",
                    len(dangling.tool_calls))
        telemetry.count("interrupted_turn_repaired", shape="buried")
        await graph.aupdate_state(config, {"messages": [RemoveMessage(id=dangling.id)]})
        return len(dangling.tool_calls)
    pending = [tc for tc in dangling.tool_calls if tc.get("id")]
    # as_node is not optional here: once a graph has more than one node LangGraph cannot
    # infer which one a write came from and raises InvalidUpdateError("Ambiguous update").
    # "tools" is the honest attribution — a ToolMessage is what that node produces. Learned
    # the hard way: the unit test's fake graph accepted the call without it, so this shipped
    # broken and only a poisoned checkpoint on the real table surfaced it.
    nodes = graph.get_graph().nodes
    telemetry.count("interrupted_turn_repaired", shape="trailing")
    await graph.aupdate_state(config, {"messages": [
        ToolMessage(content=_INTERRUPTED_TOOL, tool_call_id=tc["id"],
                    name=tc.get("name") or "tool")
        for tc in pending]}, as_node="tools" if "tools" in nodes else "model")
    return len(pending)


async def astamp_unanswered(graph, config: dict, at_ms: int) -> int:
    """Give a message whose turn ended before any model reply — stopped for consent, or
    failed — the time it arrived. _ClockMiddleware stamps a message when the model returns,
    so without this the next turn would stamp it with its own, later, arrival."""
    from langgraph.errors import InvalidUpdateError
    state = await graph.aget_state(config)
    said = []
    for m in reversed((getattr(state, "values", None) or {}).get("messages") or []):
        if isinstance(m, AIMessage):
            break
        if isinstance(m, HumanMessage) and m.id and "ts" not in m.additional_kwargs \
                and not _is_summary(m):
            said.append(m.model_copy(update={"additional_kwargs": {**m.additional_kwargs,
                                                                    "ts": at_ms}}))
    if not said:
        return 0
    update = {"messages": said[::-1]}     # same ids, so replaced in place
    try:
        await graph.aupdate_state(config, update)
    except InvalidUpdateError:
        # The last writer was not one node LangGraph can attribute the write to.
        await graph.aupdate_state(config, update, as_node="model")
    return len(said)


async def _aopen_tools(stack: AsyncExitStack, connection: dict) -> list:
    """Open an MCP session that stays alive for the session's lifetime and return its
    tools. The stack owns the teardown, so one close() releases every server."""
    session = await stack.enter_async_context(create_session(connection))
    await session.initialize()
    return await load_mcp_tools(session)


async def _abuild_session(actor_id: str, email: str, mem_sid: str,
                          workload_token: str = "", own_claim: bool = False) -> dict:
    """Build a session for this user. Agent-side 3LO: fetch the user's vaulted Lark
    token if there is one and open MCP sessions to the tool servers (SigV4 + the token
    in the custom header), kept open for the session.

    The token gates tool *calls*, not the connection: lark-mcp answers initialize and
    tools/list without it and only rejects tools/call with "authorize first" (verified
    against the deployed Runtime — an empty header returns the full tool list). So an
    unauthorized user still gets a session with the Lark tools listed, and can chat
    freely; consent is asked for when a tool is actually reached, which is the point at
    which the user can see what it is for."""
    tools: list = []
    auth_url = None
    identity_error = None
    stack = AsyncExitStack()
    # Consent is per system: each tool is tagged with the system whose server listed it, and
    # a call is walled only when *that* system has no grant. Untagged tools never wall.
    tool_idp: dict[str, str] = {}
    consent: dict[str, str] = {}   # idp → consent link; "" until a tool of it is reached
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
    for idp, (url, provider, scopes) in _DOWNSTREAMS.items():
        try:
            d_token = await asyncio.to_thread(
                lark_3lo.peek_user_token_for, actor_id, provider, scopes, workload_token)
            if not d_token:
                consent[idp] = ""
            tools += await _open(lark_3lo.mcp_connection_for(d_token, url=url), idp)
        except Exception:  # noqa: BLE001 — optional, like search
            log.exception("%s tools unavailable for %s", idp, actor_id)

    # Search doesn't depend on the user's Lark grant, so add it even when the Lark
    # tools are unavailable — an unauthorised user can still ask questions.
    if websearch.available():
        try:
            conn = await asyncio.to_thread(websearch.connection_for, actor_id)
            tools += _tag(await _aopen_tools(stack, conn), "web_search")
        except Exception:  # noqa: BLE001 — search is optional, Lark tools are not
            log.exception("web search unavailable for %s", actor_id)

    tools += _tag(_long_term_tools(actor_id), "memory")
    tools += _tag([_whoami_tool(actor_id, workload_token)], "identity")

    # Code execution, when an interpreter is deployed. The sandbox is created here rather
    # than per call so one conversation keeps one workspace session — and it is kept on the
    # cached entry so _close_session can stop it. Nothing is started yet: the session only
    # begins when the model first runs something.
    sandbox = code_tools._Sandbox(actor_id) if code_tools.available() else None
    if sandbox is not None:
        tools += _tag(code_tools.tools_for(actor_id, sandbox), "code")

    saver = _checkpointer()
    # langchain.agents.create_agent, not langgraph.prebuilt.create_react_agent: the
    # latter is deprecated as of LangGraph 1.0 and slated for removal in 2.0 (AWS's
    # devguide example still imports it). Note there are no pre/post model hooks on
    # this API — middleware replaces them — which is another reason the conversational
    # record is written at turn end instead.
    graph = create_agent(model=_model, tools=tools, system_prompt=_SYSTEM,
                         checkpointer=saver, middleware=_middleware())
    # thread_id is the whole checkpoint address (chosen by the router, so /reset can
    # rotate it) — DynamoDBSaver derives its partition key from it alone, and since it is
    # the sha256 of the actor, per-user isolation follows from the key. The agent never
    # picks it. actor_id is carried for AgentCoreMemorySaver, which rejects a config
    # without it, so switching back stays a one-line change.
    config = {"configurable": {"thread_id": mem_sid, "actor_id": actor_id}}
    # Building a session is exactly the moment a replaced microVM picks up a thread whose
    # last turn may have been killed mid tool call, so repair before the first invoke.
    try:
        repaired = await _arepair_interrupted_turn(graph, config, own_claim)
        if repaired:
            log.info("answered %d tool call(s) left by an interrupted turn", repaired)
    except Exception:  # noqa: BLE001 — a failed repair must not cost the user a session
        log.warning("could not check for an interrupted turn", exc_info=True)
    return {
        "graph": graph, "stack": stack, "sandbox": sandbox,
        "config": config,
        "actor_id": actor_id, "mem_sid": mem_sid,
        "created": time.time(),
        # auth_url is Lark's link and doubles as "identity unverified" for the web surfaces;
        # consent/tool_idp drive the per-system wall.
        "auth_url": auth_url, "identity_error": identity_error,
        "consent": consent, "tool_idp": tool_idp, "tool_source": tool_source,
        # Tool results seen during the current turn, for the auth-wall scan.
        "tool_texts": [],
    }


def _build_session(actor_id: str, email: str, mem_sid: str,
                   workload_token: str = "", own_claim: bool = False) -> dict:
    return _run(_abuild_session(actor_id, email, mem_sid, workload_token, own_claim))


def _close_session(s: dict) -> None:
    """Release everything this cached entry holds. Best-effort: a connection that is
    already gone must not stop us discarding the entry."""
    sandbox = s.get("sandbox")
    if sandbox is not None:
        sandbox.close()   # a code session left running is billed until it times out
    stack = s.get("stack")
    if stack is None:
        return
    try:
        _run(stack.aclose(), timeout=30)
    except Exception:  # noqa: BLE001
        log.warning("closing MCP sessions failed", exc_info=True)


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
        session["actor_id"], provider, scopes, workload_token)
    if kind != "auth_url":
        return ""
    session["consent"][idp] = value
    return value


_IDENTITY_ERROR = (
    "I can't reach your Lark account right now, so I have no Lark tools this turn "
    "(other questions still work).\nDetails: {err}"
)


def _get_session(actor_id: str, email: str, mem_sid: str, fresh: bool = False,
                 workload_token: str = "", own_claim: bool = False) -> dict:
    """Return the cached session for this user, rebuilding it if absent or near token
    expiry. A pending-authorization session (no token yet) is NOT cached — so the next
    turn re-checks the vault and picks up a freshly consented token."""
    cache_key = f"{actor_id}|{mem_sid}"
    log.info("session lookup: key=%s cached=%s fresh=%s",
             cache_key[-28:], cache_key in _sessions, fresh)
    with _lock:
        s = _sessions.get(cache_key)
        # `fresh` is for the consent-resume path: the user has just authorized, so a
        # cached unauthorized session must not be reused — its MCP sessions hold an
        # empty token and the turn would wall again. Measured: consent completed 31 s
        # after the prompt, well inside _UNAUTH_TTL, so waiting for expiry is not an
        # option.
        if s and fresh:
            _close_session(s)
            _sessions.pop(cache_key, None)
            s = None
        if s:
            # An unauthorized session works (tools listed, calls rejected), so it is
            # worth caching — but only briefly, or the user consents and keeps being
            # told to authorize until the full TTL lapses.
            ttl = _UNAUTH_TTL if s.get("auth_url") else _SESSION_TTL
            if (time.time() - s["created"]) < ttl:
                return s
            _close_session(s)
        s = _build_session(actor_id, email, mem_sid, workload_token, own_claim)
        # One retry when the vault says "not consented". GetResourceOauth2Token is
        # nondeterministic for a grant that exists: measured in one container, seconds apart,
        # identical arguments — one build got the token, the next only an authorizationUrl.
        # Without this a consented user is told to authorise again at random.
        if s.get("auth_url") and workload_token:
            retry = _build_session(actor_id, email, mem_sid, workload_token, own_claim)
            # Keep the retry either way. Each attempt mints a pending consent session and
            # invalidates the one before it, so handing back the first attempt's URL gave the
            # user a link that this retry had already killed — every completion then failed
            # with "Invalid or expired session". Only the newest URL is live.
            _close_session(s)
            s = retry
        if not s.get("identity_error"):
            _sessions[cache_key] = s
        return s


def chat_result(actor_id: str, message: str, email: str = "",
                mem_sid: str = "", workload_token: str = "") -> dict:
    """Non-streaming chat → {reply, needs_auth, auth_url}. History via Memory.
    When the user hasn't authorized Lark yet, needs_auth is True and auth_url is the
    raw consent URL, so the caller (router) can drive the wait-for-consent loop instead
    of asking the user to re-send."""
    received_ms = int(time.time() * 1000)   # the user's message is stamped with this
    remember_thread(actor_id, mem_sid)
    thread_id = mem_sid or thread_for(actor_id)
    # A turn is already running on this thread. This surface answers synchronously and the
    # running turn's answer appears on another surface, so steering the message there would
    # promise an answer this caller never receives. It is told to send again instead.
    owner = try_claim(thread_id, "sync")
    if owner is not None:
        telemetry.count("busy_elsewhere", surface="sync")
        return {"reply": _BUSY_ELSEWHERE.get(owner, _BUSY_ELSEWHERE["sync"]),
                "needs_auth": False, "busy": owner}
    try:
        s = _get_session(actor_id, email, thread_id, workload_token=workload_token,
                         own_claim=True)
        if s.get("identity_error"):
            # Answer without tools, but say so — silently degrading is what made a missing
            # IAM permission look like the model choosing not to help.
            return {"reply": _IDENTITY_ERROR.format(err=s["identity_error"]),
                    "needs_auth": False, "identity_error": s["identity_error"]}
        s["tool_texts"] = []
        turn_id = "msg-" + uuid.uuid4().hex
        with telemetry.timed("turn", surface="sync"):
            state = _run(s["graph"].ainvoke(
                {"messages": [HumanMessage(message, id=turn_id)]},
                config=_run_config(s, delivery.NoDelivery("this channel"), received_ms)))
    finally:
        release(thread_id)
    # This turn's tool results only. The state is the whole thread, so scanning all of it
    # let one old "no user token" result raise a consent prompt on every later sync turn.
    messages = state.get("messages", [])
    start = next((i for i, m in enumerate(messages) if getattr(m, "id", None) == turn_id), 0)
    for m in messages[start:]:
        if isinstance(m, ToolMessage):
            s["tool_texts"].append((m.name, _message_text(m)))
    reply = _message_text(state["messages"][-1]) if state.get("messages") else ""
    idp = _hit_auth_wall(s)
    link = _consent_link(s, idp, workload_token) if idp else ""
    if link:
        return {"reply": _AUTH_PROMPT.format(label=idp.capitalize(), url=link),
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
    remember_thread(actor_id, mem_sid)
    thread_id = mem_sid or thread_for(actor_id)
    handles = list(image_handles or [])
    blocks = images.prepare(handles)        # network, so before the lock

    # One step under the lock: see who holds the thread, then claim it, steer into it, or
    # step aside. A separate check and claim left a cold session build between them, so two
    # turns could start on one thread. On the same surface the message is steered into the
    # running turn, whose answer appears where the user is looking; on another surface it
    # would be answered in that window, so it is told instead — and registers nothing, so
    # it cannot join that turn's attachments.
    with _in_flight_lock:
        owner = _in_flight_threads.get(thread_id)
        # The running turn answers into another chat — a DM and a group share one thread.
        if owner == "lark" and _claim_dest.get(thread_id) != chat_id:
            owner = "lark-other"
        if owner in (None, "lark"):
            if owner is None:
                _track(+1, thread_id, "lark", chat_id)
            marks, turn_message_id = images.register(thread_id, handles, blocks)
            if marks:
                message = "\n".join([message, marks]).strip() if message else marks
            if owner == "lark":
                steer(thread_id, message, turn_message_id, fresh=fresh_session)
                if message_id and not fresh_session:
                    _steer_reply_to[thread_id] = message_id
    if owner == "lark":
        telemetry.count("steered", surface="lark")
        _quietly(lark_notify.remove_reaction, message_id, reaction_id)
        return {"accepted": True, "needs_auth": False, "steered": True}
    if owner is not None:
        telemetry.count("busy_elsewhere", surface="lark")
        _quietly(lark_notify.remove_reaction, message_id, reaction_id)
        lark_notify.send_text(chat_id, _BUSY_ELSEWHERE.get(owner, _BUSY_ELSEWHERE["sync"]))
        return {"accepted": True, "needs_auth": False, "busy": owner}

    # The thread is claimed from here on. Every way out has to release it, or the thread
    # reads as busy for good and every later message is steered into a turn that is gone.
    try:
        s = _get_session(actor_id, email, thread_id, fresh=fresh_session,
                         workload_token=workload_token, own_claim=True)
        if s.get("identity_error"):
            _release_unanswered(thread_id, chat_id)
            return {"reply": _IDENTITY_ERROR.format(err=s["identity_error"]),
                    "needs_auth": False, "identity_error": s["identity_error"]}

        def _answer(s: dict, inputs: list) -> None:
            try:
                with telemetry.timed("turn", surface="lark"):
                    _stream_to_chat(s, inputs, chat_id, received_ms=received_ms)
                idp = _hit_auth_wall(s)
                if idp:
                    telemetry.count("auth_wall", surface="lark", idp=idp)
                    link = _consent_link(s, idp, workload_token)
                    if link:
                        # The message was parked before this turn, so the shim's /return
                        # replays it once consent lands — no re-send.
                        lark_notify.send_link(
                            chat_id, f"需要访问你的 {idp.capitalize()} 账号，授权后我会自动继续：",
                            "点击授权", link)
                    else:
                        _drop_sessions(actor_id)
                        lark_notify.send_text(chat_id, f"{idp.capitalize()} 已授权，请再发一次。")
            except Exception as e:  # noqa: BLE001 — the caller is already gone
                log.exception("async turn failed for %s", actor_id)
                # This session stays cached, so a turn that died mid tool call would make
                # every later turn fail too. Repair here as well as at build time.
                try:
                    _run(_arepair_interrupted_turn(s["graph"], s["config"], own_claim=True),
                         timeout=30)
                except Exception:  # noqa: BLE001
                    log.warning("could not repair the interrupted turn", exc_info=True)
                _quietly(lark_notify.send_text, chat_id,
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
                    with _in_flight_lock:
                        inputs = _take_steer(thread_id)
                        if not inputs:
                            release(thread_id)
                            released = True
                    if inputs:
                        telemetry.count("steer_continued")
                        # Fetched again rather than reused: the consent branch above may
                        # have dropped this session, and a consent replay among the inputs
                        # needs one built after the grant, not the unauthorised one.
                        fresh = any(item[2] for item in inputs)
                        if fresh:
                            _drop_sessions(actor_id)
                        # These inputs are already off the queue, so a failure here has to
                        # count them itself — the release at the end only sees what is left.
                        try:
                            session = _get_session(actor_id, email, thread_id, fresh=fresh,
                                                   workload_token=workload_token, own_claim=True)
                            problem = session.get("identity_error")
                        except Exception as e:  # noqa: BLE001 — told below, not raised
                            log.exception("continuation session failed for %s", actor_id)
                            problem = type(e).__name__
                        if problem:
                            _quietly(lark_notify.send_text, chat_id,
                                     f"还有 {len(inputs)} 条消息没来得及处理（{problem}），请重新发送。")
                            break
            finally:
                if not released:
                    _release_unanswered(thread_id, chat_id)
                # Last, and on its own: it is a network call that can raise, and the claim
                # must be released whatever it does. A marker left behind is cosmetic.
                _quietly(lark_notify.remove_reaction, message_id, reaction_id)

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


_STREAM_END = object()
# Marks a queued call for the consumer to make, rather than text to yield.
_EVENT = object()


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


def _turn_input(message, message_id: str | None = None) -> list:
    """The HumanMessages a turn starts from: one string, or (text, id, fresh, ms) entries —
    several when a continuation answers what was steered in after the last model call."""
    if isinstance(message, str):
        return [HumanMessage(message, id=message_id)]
    return [_said(item) for item in message]


def _said(item: tuple) -> HumanMessage:
    """A queued (text, id, fresh[, arrived ms]) entry as the message it becomes, carrying
    when it arrived: a continuation answers it later than that."""
    ts = item[3] if len(item) > 3 else None
    return HumanMessage(item[0], id=item[1] or None,
                        additional_kwargs={"ts": ts} if ts else {})


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
            q.put(_STREAM_END)

    asyncio.run_coroutine_threadsafe(_pump(), _LOOP)
    while True:
        item = q.get()
        if item is _STREAM_END:
            return
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple) and item[0] is _EVENT:
            item[1](*item[2:])
            continue
        yield item


async def arun_in_thread(fn, *args):
    """Run a blocking call off the caller's event loop."""
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


async def aget_session(actor_id: str, workload_token: str = "", thread: str = "",
                       own_claim: bool = False) -> dict:
    """The cached session for this actor, fetched without blocking the caller's loop.
    Session building opens MCP connections, which is seconds of I/O. `thread` is one the
    caller named and server.py checked belongs to the actor; else the one last named.
    `own_claim` when the caller holds the thread's claim, so the build still repairs it."""
    return await arun_in_thread(_get_session, actor_id, "",
                               thread or thread_for(actor_id), False, workload_token,
                               own_claim)


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


def _stream_to_chat(session: dict, message, chat_id: str,
                    message_id: str | None = None, received_ms: int | None = None) -> str:
    """Run the turn, typing the answer into a streaming card as it is produced, and
    return the full text. If the card can't be created or an update fails (e.g. the
    cardkit:card:write scope isn't granted), fall back to posting the final text — the
    answer must survive regardless. Consent replies are handled by the caller."""
    session["tool_texts"] = []
    # One card per stretch of the turn: a message steered in mid-turn closes the current
    # card and the rest is answered on a new one, below that message.
    cards = [lark_notify.StreamingCard(chat_id)]
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
            if not lark_notify.send_text(chat_id, text):
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
                group = lark_notify.is_group(chat_id)
            # Quoted in a group, so others can see which message is being answered.
            cards.append(lark_notify.StreamingCard(
                chat_id, reply_to=reply_to if group else "", carry=cards[-1]))
            streaming = cards[-1].open()
        except Exception:  # noqa: BLE001 — the rest of the answer must still arrive
            log.warning("could not hand the turn over to a new card", exc_info=True)
            streaming = False

    thread_id = (session.get("config") or {}).get("configurable", {}).get("thread_id", "")
    events: queue.Queue = queue.Queue()
    in_order = _InStreamOrder(events, lambda: cards[-1], _on_steered)
    if thread_id:
        _turn_cards[thread_id] = in_order
    try:

        # A tool whose system has no grant is certain to be refused, and letting the turn run
        # on means the model narrates that refusal at length before the consent card arrives.
        # Stop at the tool call instead — the caller sees `walled_tool` and posts the card.
        def _abort_on_ungranted_tool(tool_name: str) -> bool:
            idp = _walled_idp(session, tool_name)
            if not idp:
                return False
            session["walled_tool"] = tool_name
            log.info("aborting turn: %s needs %s consent", tool_name, idp)
            return True

        def _on_tool_result(name: str, text: str) -> None:
            session["tool_texts"].append((name, text))

        for delta in _iter_deltas(session["graph"], message,
                                  _run_config(session, delivery.LarkDelivery(chat_id),
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
            idp = _walled_idp(session, session["walled_tool"]) or "lark"
            text = f"🔐 这一步需要访问你的 {idp.capitalize()} 账号"

        _deliver(text, last=True)
        return "".join(earlier) + text
    except BaseException:
        # The stream failed: stop the card claiming it is still working.
        if streaming:
            _quietly(cards[-1].close, "".join(acc))
        raise
    finally:
        # Only this turn's card: a later turn may already have registered its own.
        if _turn_cards.get(thread_id) is in_order:
            _turn_cards.pop(thread_id, None)


def _thread_config(actor_id: str, mem_sid: str) -> dict:
    return {"configurable": {"thread_id": mem_sid or _session_id_for(actor_id),
                             "actor_id": actor_id}}


def history_stats(actor_id: str, mem_sid: str = "") -> dict:
    """What this thread holds → {"turns": n, "toolCalls": n, "messages": n}.

    Turns, not messages, because a message count is an implementation detail: LangGraph
    splits state in a way the user cannot see, and unevenly — a turn that calls a tool
    produces more messages than one that does not. A turn here is an *answer delivered*
    (an AIMessage with no pending tool_calls), which is the unit a person can check against
    their own chat window, and which a summary cannot inflate (the injected summary is a
    HumanMessage).

    toolCalls is reported alongside because a good-looking answer is not evidence that a
    tool ran — that mistake cost a long debugging session. It says a call was made, not
    whose identity made it; only the MCP server's own log answers that.

    Answered here rather than in the router because reading the conversation means decoding
    a checkpoint, and agent_core is the only module that knows the framework."""
    try:
        tup = _run(_checkpointer().aget_tuple(_thread_config(actor_id, mem_sid)),
                   timeout=30)
    except Exception:  # noqa: BLE001 — /status must answer even if this part cannot
        log.warning("could not read thread stats", exc_info=True)
        return {"turns": 0, "toolCalls": 0, "messages": 0, "unavailable": True}
    values = (tup.checkpoint.get("channel_values") if tup else None) or {}
    messages = values.get("messages") or []
    ai = [m for m in messages if isinstance(m, AIMessage)]
    # The context size is what Bedrock billed for the latest model call, read from that
    # call's own usage — not an estimate, which matters because the library's estimator
    # charges a flat 85 tokens per image. input_tokens already includes the cached part.
    usage = next((m.usage_metadata for m in reversed(ai)
                  if getattr(m, "usage_metadata", None)), None) or {}
    # Every model call in the thread, for the cache hit rate over the conversation. Only
    # what the checkpoint still holds: a compaction drops the older calls with their turns.
    calls = [m.usage_metadata for m in ai if getattr(m, "usage_metadata", None)]
    return {
        "turns": sum(1 for m in ai if not (getattr(m, "tool_calls", None) or [])),
        "toolCalls": sum(len(getattr(m, "tool_calls", None) or []) for m in ai),
        "messages": sum(1 for m in messages
                        if isinstance(m, (HumanMessage, AIMessage))),
        "contextTokens": usage.get("input_tokens", 0),
        "cachedTokens": (usage.get("input_token_details") or {}).get("cache_read", 0),
        "modelCalls": len(calls),
        "inputTotal": sum(u.get("input_tokens", 0) for u in calls),
        "cachedTotal": sum((u.get("input_token_details") or {}).get("cache_read", 0)
                           for u in calls),
    }


_TRANSCRIPT_LIMIT = int(os.environ.get("TRANSCRIPT_LIMIT", "60"))
_TRANSCRIPT_TOOL_CHARS = 800


def transcript(actor_id: str, mem_sid: str = "") -> dict:
    """This thread's messages, render-ready → {"messages": [...], "truncated": bool}.

    Lets a reloaded page show the conversation the agent actually remembers rather than a
    copy the tab was keeping: after summarisation or /clear the two would disagree, and the
    checkpoint is the one that decides the next turn."""
    try:
        tup = _run(_checkpointer().aget_tuple(_thread_config(actor_id, mem_sid)), timeout=30)
    except Exception:  # noqa: BLE001 — a blank page beats a broken one
        log.warning("could not read transcript", exc_info=True)
        return {"messages": [], "unavailable": True}
    values = (tup.checkpoint.get("channel_values") if tup else None) or {}
    history = values.get("messages") or []
    tail = history[-_TRANSCRIPT_LIMIT:]
    out = []
    for m in tail:
        if _is_summary(m):
            # Written by summarisation, not by the user; the page shows it as a note.
            out.append({"role": "summary", "content": _text_of(m)})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": _text_of(m), **_said_at(m)})
        elif isinstance(m, AIMessage):
            for call in getattr(m, "tool_calls", None) or []:
                out.append({"role": "toolCall", **_said_at(m), "id": call.get("id") or "",
                            "name": call.get("name") or "tool",
                            "args": json.dumps(call.get("args") or {}, ensure_ascii=False)})
            if _text_of(m):
                out.append({"role": "assistant", "content": _text_of(m), **_said_at(m)})
        elif isinstance(m, ToolMessage):
            out.append({"role": "toolResult", "id": getattr(m, "tool_call_id", "") or "",
                        "content": _text_of(m)[:_TRANSCRIPT_TOOL_CHARS]})
            sent = _delivered(m)
            if sent:
                # Signed afresh on every read. A failed signature costs the link, not the
                # history: the file is still named.
                try:
                    link = deliveries.url(sent["key"], sent["name"], actor_id)
                except Exception:  # noqa: BLE001
                    log.warning("could not sign a delivered file's link", exc_info=True)
                    link = ""
                out.append({"role": "file", "name": sent["name"], "size": sent.get("size", 0),
                            "url": link})
    return {"messages": out, "truncated": len(history) > len(tail)}


def _delivered(message) -> dict | None:
    """The stored file a deliver_file result carries on its artifact, if any."""
    artifact = getattr(message, "artifact", None)
    sent = artifact.get("file") if isinstance(artifact, dict) else None
    return sent if isinstance(sent, dict) and sent.get("key") and sent.get("name") else None


def _said_at(message) -> dict:
    """{"ts": epoch ms} as _ClockMiddleware stamped it; {} for messages older than it."""
    ts = (getattr(message, "additional_kwargs", None) or {}).get("ts")
    return {"ts": ts} if isinstance(ts, int) else {}


def _text_of(message) -> str:
    """Content is a string for simple turns and a list of blocks for multimodal ones."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts = [b.get("text", "") for b in content or [] if isinstance(b, dict)]
    return "".join(p for p in parts if p)


def clear_history(actor_id: str, mem_sid: str = "") -> dict:
    """Delete this thread's stored state → {"deleted": bool}.

    Really deletes, unlike /reset which rotates to a new thread and leaves the old one
    readable. Also drops the cached session, because it holds a compiled graph whose next
    turn would otherwise write on top of a thread the user believes is gone.

    Long-term memory is NOT touched yet: no Memory resource exists, so there is nothing
    to delete. When one does, its records belong in here too — a user asking to clear
    history does not mean "except the parts extracted from it"."""
    config = _thread_config(actor_id, mem_sid)
    thread_id = config["configurable"]["thread_id"]
    with _lock:
        for key in [k for k in _sessions if k.endswith(f"|{thread_id}")]:
            s = _sessions.pop(key, None)
            if s:
                _close_session(s)
    # The files it delivered go with it; read before the thread is, as nothing else records
    # their keys. Every checkpoint, not the latest: a summary removes older messages from
    # the latest, and their files with them from view.
    try:
        async def keys() -> list[str]:
            found = []
            async for tup in _checkpointer().alist(config):
                values = (tup.checkpoint or {}).get("channel_values") or {}
                found += [f["key"] for f in map(_delivered, values.get("messages") or []) if f]
            return found
        deliveries.delete(_run(keys(), timeout=120), actor_id)
    except Exception:  # noqa: BLE001 — the lifecycle rule removes what this misses
        log.warning("could not remove delivered files of %s", thread_id, exc_info=True)
    try:
        _run(_checkpointer().adelete_thread(thread_id), timeout=60)
    except Exception as e:  # noqa: BLE001
        log.warning("could not clear thread %s", thread_id, exc_info=True)
        return {"deleted": False, "error": type(e).__name__}
    return {"deleted": True}


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
    with ThreadPoolExecutor(max_workers=max(1, len(held))) as pool:
        found = {idp: pool.submit(_ACCOUNT_OF[idp], token) for idp, token in held.items()}
    status["accounts"] = {idp: f.result() for idp, f in found.items() if f.result()}
    return status


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
            "which systems (Lark, Google) they have authorised and with which account. Call "
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


# Who signed in to grant each system's token, read with that token: shown by /auth.
_ACCOUNT_OF = {"lark": lark_3lo.account_of, "google": _google_account}


def _drop_sessions(actor_id: str) -> None:
    """Drop every cached session for this user so the next turn re-reads the vault."""
    with _lock:
        for key in [k for k in _sessions if k.startswith(f"{actor_id}|")]:
            s = _sessions.pop(key, None)
            if s:
                _close_session(s)


def reauth(actor_id: str, idp: str = "lark", workload_token: str = "") -> dict:
    """Start a fresh 3LO flow for `idp` even when a token is already vaulted → {auth_url}.

    Per-IdP, and the Lark path stays distinct for a reason: only it can ask the vendor whose
    token this actually is and refuse a mismatch. A second system reuses the generic vault
    fetch, which cannot make that check — see lark_3lo.get_user_token_for."""
    lark_user = actor_id.startswith("lark:") and lark_3lo.configured()
    idp = idp or ("lark" if lark_user else "")
    if idp == "lark" and not lark_3lo.configured():
        return {"reply": "这个部署没有接入 Lark。", "needs_auth": False}
    if idp != "lark" and idp not in _DOWNSTREAMS:
        # The router lists what it has registered; the agent may lack the server or provider.
        known = idp == "google"
        return {"reply": f"{idp.capitalize()} 未配置（需要 OAuth provider 与已部署的 MCP server）"
                if known else f"IdP 尚未接入：{idp}", "needs_auth": False}
    _drop_sessions(actor_id)
    if idp == "lark" and lark_user:
        kind, value = lark_3lo.get_user_lark_token(actor_id, force=True,
                                                   workload_token=workload_token)
    elif idp == "lark":
        provider, scopes = lark_3lo.lark_provider()
        kind, value = lark_3lo.get_user_token_for(actor_id, provider, scopes, workload_token,
                                                  force=True)
    else:
        _, provider, scopes = _DOWNSTREAMS[idp]
        kind, value = lark_3lo.get_user_token_for(actor_id, provider, scopes, workload_token,
                                                  force=True)
    if kind == "auth_url":
        return {"reply": _AUTH_PROMPT.format(label=idp.capitalize(), url=value),
                "needs_auth": True, "auth_url": value}
    return {"reply": "Already authorized.", "needs_auth": False}
