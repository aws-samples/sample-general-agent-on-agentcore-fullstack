"""Server-reuse model: the model, the MCP sessions and the compiled graph are built ONCE
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
    tools in tools/memory.py. Nothing is written per turn: retrieval is the priced operation,
    so storing every exchange and querying blindly is the pattern that makes cost scale
    with chat volume instead of with need. What to keep is the model's explicit decision.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from contextlib import AsyncExitStack

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from channels import web_files
from identity import lark_3lo
from runtime import telemetry
import core
from identity import grants
import middleware
import tools
from tools import browser
from runtime import turns

log = logging.getLogger("agent.sessions")

_CHECKPOINT_TABLE = os.environ.get("CHECKPOINT_TABLE", "")


_CHECKPOINT_BUCKET = os.environ.get("CHECKPOINT_BUCKET", "")


# One thread per user, rotated only by /reset, so the state has to expire on its own.
_CHECKPOINT_TTL = int(os.environ.get("CHECKPOINT_TTL_DAYS", "365")) * 86400


# Rebuild a cached session before its Cognito access token (~1h) expires.
_SESSION_TTL = int(os.environ.get("SESSION_TTL_SECONDS", "3000"))  # 50 min


# An unauthorized session is cached only briefly: it works, but the moment the user
# consents we want the next turn to pick up the token.
_UNAUTH_TTL = int(os.environ.get("UNAUTH_SESSION_TTL_SECONDS", "60"))


# session_id -> {graph, config, stack, created, ...}. One microVM ≈ one session.
_sessions: dict[str, dict] = {}


_lock = threading.Lock()


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
    if turns.thread_busy(thread_id):
        return {"compacted": 0, "reason": "busy"}
    session = _get_session(actor_id, "", thread_id, workload_token=workload_token)
    if session.get("identity_error"):
        return {"compacted": 0, "reason": "identity"}
    return turns._run(_acompact(session["graph"], session["config"]),
                      timeout=middleware._SUMMARY_TIMEOUT + 30)


async def _acompact(graph, config) -> dict:
    messages = (await graph.aget_state(config)).values.get("messages") or []
    summariser = middleware._summarization(core._summary_model(), trigger=("messages", 1),
                                           keep=("messages", _COMPACT_KEEP))
    # Asked first so "nothing to compact" is not reported as a failed summary. The
    # cutoff is what the middleware itself would use; langchain is pinned <2 for it.
    if summariser._determine_cutoff_index(messages) <= 0:
        return {"compacted": 0, "reason": "short", "messages": len(messages)}
    update = await summariser.abefore_model({"messages": messages}, None)
    if not update:
        return {"compacted": 0, "reason": "failed"}
    await graph.aupdate_state(config, update)
    after = (await graph.aget_state(config)).values.get("messages") or []
    kept = sum(1 for m in after if not middleware._is_summary(m))
    telemetry.count("compacted")
    return {"compacted": len(messages) - kept, "kept": kept}


async def atools_overview(actor_id: str, mem_sid: str, workload_token: str = "") -> dict:
    """tools_overview for the session chat would use — the same cache key, so asking does
    not assemble a second session and handshake every server again."""
    session = await turns.arun_in_thread(_get_session, actor_id, "",
                                         mem_sid or thread_for(actor_id), False, workload_token)
    return tools_overview(session)


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
        region_name=core._REGION,
        ttl_seconds=_CHECKPOINT_TTL,
        # Compress first so spilling to S3 stays the exception: DynamoDB caps an item at
        # 400 KB, and one unbounded thread per user would otherwise reach it.
        enable_checkpoint_compression=True,
        s3_offload_config=(
            {"bucket_name": _CHECKPOINT_BUCKET, "key_prefix": "state"}
            if _CHECKPOINT_BUCKET else None),
    )


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
        if turns.thread_busy(config["configurable"]["thread_id"]) and not own_claim:
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
                and not middleware._is_summary(m):
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


async def _abuild_session(actor_id: str, email: str, mem_sid: str,
                          workload_token: str = "", own_claim: bool = False) -> dict:
    """Build a session for this user: its tools (tools.assemble), the agent over them
    (core.build), and the conversation's checkpoint, repaired if its last turn was cut off."""
    stack = AsyncExitStack()
    kit = await tools.assemble(actor_id, stack, workload_token, thread_id=mem_sid)
    graph = core.build(kit["tools"], _checkpointer())
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
        "graph": graph, "stack": stack, "sandbox": kit["sandbox"],
        "config": config,
        "actor_id": actor_id, "mem_sid": mem_sid,
        "created": time.time(),
        # auth_url is Lark's link and doubles as "identity unverified" for the web surfaces;
        # consent/tool_idp drive the per-system wall.
        "auth_url": kit["auth_url"], "identity_error": kit["identity_error"],
        "consent": kit["consent"], "refused": kit["refused"], "tool_idp": kit["tool_idp"],
        "tool_source": kit["tool_source"],
        # Tool results seen during the current turn, for the auth-wall scan.
        "tool_texts": [],
    }


def _build_session(actor_id: str, email: str, mem_sid: str,
                   workload_token: str = "", own_claim: bool = False) -> dict:
    return turns._run(_abuild_session(actor_id, email, mem_sid, workload_token, own_claim))


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
        turns._run(stack.aclose(), timeout=30)
    except Exception:  # noqa: BLE001
        log.warning("closing MCP sessions failed", exc_info=True)


def _get_session(actor_id: str, email: str, mem_sid: str, fresh: bool = False,
                 workload_token: str = "", own_claim: bool = False) -> dict:
    """Return the cached session for this user, rebuilding it if absent or near token
    expiry. A pending-authorization session (no token yet) is NOT cached — so the next
    turn re-checks the vault and picks up a freshly consented token."""
    cache_key = f"{actor_id}|{mem_sid}"
    if browser.available():
        browser.sweep()      # browsers outlive sessions, so idle ones are stopped from here
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


async def aget_session(actor_id: str, workload_token: str = "", thread: str = "",
                       own_claim: bool = False) -> dict:
    """The cached session for this actor, fetched without blocking the caller's loop.
    Session building opens MCP connections, which is seconds of I/O. `thread` is one the
    caller named and server.py checked belongs to the actor; else the one last named.
    `own_claim` when the caller holds the thread's claim, so the build still repairs it."""
    return await turns.arun_in_thread(_get_session, actor_id, "",
                                      thread or thread_for(actor_id), False, workload_token,
                                      own_claim)


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
    a checkpoint, which needs the agent's framework."""
    try:
        tup = turns._run(_checkpointer().aget_tuple(_thread_config(actor_id, mem_sid)),
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
        tup = turns._run(_checkpointer().aget_tuple(_thread_config(actor_id, mem_sid)), timeout=30)
    except Exception:  # noqa: BLE001 — a blank page beats a broken one
        log.warning("could not read transcript", exc_info=True)
        return {"messages": [], "unavailable": True}
    values = (tup.checkpoint.get("channel_values") if tup else None) or {}
    history = values.get("messages") or []
    tail = history[-_TRANSCRIPT_LIMIT:]
    out = []
    for m in tail:
        if middleware._is_summary(m):
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
                    link = web_files.url(sent["key"], sent["name"], actor_id)
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
    if browser.available():
        browser.close(actor_id, thread_id)
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
        web_files.delete(turns._run(keys(), timeout=120), actor_id)
    except Exception:  # noqa: BLE001 — the lifecycle rule removes what this misses
        log.warning("could not remove delivered files of %s", thread_id, exc_info=True)
    try:
        turns._run(_checkpointer().adelete_thread(thread_id), timeout=60)
    except Exception as e:  # noqa: BLE001
        log.warning("could not clear thread %s", thread_id, exc_info=True)
        return {"deleted": False, "error": type(e).__name__}
    return {"deleted": True}


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
    if idp != "lark" and idp not in grants._DOWNSTREAMS:
        # The router lists what it has registered; the agent may lack the server or provider.
        need = {"google": "OAuth provider 与已部署的 MCP server", "github": "OAuth provider，并重新部署 runtime"}
        return {"reply": f"{grants._label(idp)} 未配置（需要 {need[idp]}）"
                if idp in need else f"IdP 尚未接入：{idp}", "needs_auth": False}
    _drop_sessions(actor_id)
    if idp == "lark" and lark_user:
        kind, value = lark_3lo.get_user_lark_token(actor_id, force=True,
                                                   workload_token=workload_token)
    elif idp == "lark":
        provider, scopes = lark_3lo.lark_provider()
        kind, value = lark_3lo.get_user_token_for(actor_id, provider, scopes, workload_token,
                                                  force=True)
    else:
        _, provider, scopes = grants._DOWNSTREAMS[idp]
        kind, value = lark_3lo.get_user_token_for(actor_id, provider, scopes, workload_token,
                                                  force=True)
    if kind == "auth_url":
        return {"reply": grants._AUTH_PROMPT.format(label=grants._label(idp), url=value),
                "needs_auth": True, "auth_url": value}
    return {"reply": "Already authorized.", "needs_auth": False}
