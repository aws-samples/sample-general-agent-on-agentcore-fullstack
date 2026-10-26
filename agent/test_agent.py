"""Unit tests for agent logic that doesn't require live AWS.

Run: tests/run.sh (which supplies the deps), or from this directory:
  uv run --with langchain --with langchain-aws --with langgraph \
         --with langchain-mcp-adapters --with boto3 --with httpx --with pytest \
         python -m pytest test_agent.py -v

`agent_core` is imported for real. That was impossible under Strands, whose wheels
target the ARM64 runtime, so the streaming tests used to exec slices of the source with
a faked `_iter_deltas` — asserting against a copy of the code rather than the code. The
LangGraph stack is pure Python, so the turn loop is now driven through an actual
compiled graph with a scripted model. Constructing the model touches no AWS
credentials; nothing here calls Bedrock.
"""

import base64
import hashlib
import json
import time
import os
import re
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(__file__))
# What a deploy injects; the modules read it at import and have no fallback.
os.environ.setdefault("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-5")
os.environ.setdefault("LARK_API_DOMAIN", "https://open.larksuite.com")
os.environ.setdefault("LARK_OAUTH_PROVIDER", "test-3lo")
os.environ.setdefault("AGENT_WORKLOAD_NAME", "test-wl")
os.environ.setdefault("METRICS_NAMESPACE", "test/agent")

import agent_core

# Message classes are used throughout; agent_core already pulls them in.
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ------------------------------- web search ----------------------------------

def test_web_search_is_signed_by_the_agent_role_not_a_users_token():
    """Search reads no user data, and minting a per-user token for it would need the agent
    to hold Cognito admin rights — and create a Cognito user for every web actor."""
    import websearch
    with mock.patch.object(websearch, "_GATEWAY_URL",
                           "https://gw-1.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"):
        conn = websearch.connection_for("web:abc")
    assert "Authorization" not in conn["headers"]
    assert conn["auth"]._signer._region_name == "us-east-1"
    assert conn["auth"]._signer._service_name == "bedrock-agentcore"


# ------------------------------- agent_core session id ----------------------

def test_session_id_deterministic_per_user():
    """Load just the _session_id_for function without importing the heavy deps."""
    src = open(os.path.join(os.path.dirname(__file__), "agent_core.py"), encoding="utf-8").read()
    ns = {"hashlib": hashlib}
    # exec only the function definition we care about
    start = src.index("def _session_id_for")
    end = src.index("\n\ndef ", start)
    exec(src[start:end], ns)  # nosec B102 # nosemgrep: exec-detected — asserts a source contract
    sid = ns["_session_id_for"]
    assert sid("lark:ou_abc") == sid("lark:ou_abc")          # stable
    assert sid("lark:ou_abc") != sid("lark:ou_xyz")          # per-user
    assert sid("lark:ou_abc").startswith("sess-")


# ------------------------------- checkpointer -------------------------------
# The conversation lives in the checkpointer, so the two ways this can go wrong are both
# silent: picking the fallback in a real deployment (history dies with the microVM, which
# reads as "the bot forgot everything") or mis-wiring the offload (a long conversation
# hits DynamoDB's 400 KB item cap mid-turn).

def test_checkpointer_falls_back_to_one_shared_in_process_saver():
    """No table configured — a session must still work, and the fallback has to be shared:
    a fresh InMemorySaver per session build would drop history every cache expiry."""
    with mock.patch.object(agent_core, "_CHECKPOINT_TABLE", ""), \
         mock.patch.object(agent_core, "_fallback_saver", None):
        first = agent_core._checkpointer()
        second = agent_core._checkpointer()
    from langgraph.checkpoint.memory import InMemorySaver
    assert isinstance(first, InMemorySaver)
    assert first is second


def test_checkpointer_uses_dynamodb_with_offload_when_configured():
    with mock.patch.object(agent_core, "_CHECKPOINT_TABLE", "tbl"), \
         mock.patch.object(agent_core, "_CHECKPOINT_BUCKET", "bkt"), \
         mock.patch("langgraph_checkpoint_aws.DynamoDBSaver") as saver:
        agent_core._checkpointer()
    kwargs = saver.call_args.kwargs
    assert kwargs["table_name"] == "tbl"
    assert kwargs["ttl_seconds"] == agent_core._CHECKPOINT_TTL
    # Compression keeps the spill rare; the spill keeps a long thread from failing hard.
    assert kwargs["enable_checkpoint_compression"] is True
    assert kwargs["s3_offload_config"]["bucket_name"] == "bkt"


def test_checkpointer_omits_offload_when_no_bucket():
    """Passing a config with an empty bucket name would fail at write time, deep inside
    the saver, instead of simply not offloading."""
    with mock.patch.object(agent_core, "_CHECKPOINT_TABLE", "tbl"), \
         mock.patch.object(agent_core, "_CHECKPOINT_BUCKET", ""), \
         mock.patch("langgraph_checkpoint_aws.DynamoDBSaver") as saver:
        agent_core._checkpointer()
    assert saver.call_args.kwargs["s3_offload_config"] is None


def _repair(messages):
    """Drive the real repair against a graph whose state is `messages`, returning the
    count and whatever it appended."""
    import asyncio
    appended = []

    class _Graph:
        """Mimics the real graph's contract, including the part a looser fake hid: a
        multi-node graph rejects aupdate_state without as_node."""

        def get_graph(self):
            return type("G", (), {"nodes": ["__start__", "model", "tools", "__end__"]})()

        async def aget_state(self, config):
            return type("S", (), {"values": {"messages": messages}})()

        async def aupdate_state(self, config, values, as_node=None):
            # A removal needs no attribution; an appended ToolMessage does, and the real
            # graph raises without it.
            from langchain_core.messages import RemoveMessage
            if as_node is None and not all(isinstance(m, RemoveMessage)
                                           for m in values["messages"]):
                raise RuntimeError("Ambiguous update, specify as_node")
            if as_node:
                appended.append(as_node)
            appended.extend(values["messages"])

    config = {"configurable": {"thread_id": "t-test"}}
    n = asyncio.run(agent_core._arepair_interrupted_turn(_Graph(), config))
    return n, appended


def test_interrupted_tool_call_is_answered_so_the_thread_stays_usable():
    """Bedrock rejects a toolUse with no toolResult, so without this the whole thread
    becomes permanently unusable after one killed turn — history intact, unreachable."""
    from langchain_core.messages import AIMessage as AI, HumanMessage
    dangling = AI(content="", tool_calls=[
        {"name": "lark_list_my_docs", "args": {}, "id": "call-1"}])
    n, appended = _repair([HumanMessage("hi"), dangling])
    assert n == 1
    # The write must be attributed to the tools node or LangGraph refuses it outright.
    assert appended[0] == "tools"
    assert appended[1].tool_call_id == "call-1"


def test_a_buried_dangling_call_is_dropped_rather_than_answered_out_of_position():
    """The shape that actually wedged a live thread. LangGraph writes the new HumanMessage
    before calling the model, so a failed turn buries the dangling AIMessage one deeper
    every time — and appending cannot fix it, since a toolResult must come *immediately*
    after its toolUse. Dropping the message is the only legal repair there."""
    from langchain_core.messages import AIMessage as AI, HumanMessage, RemoveMessage
    dangling = AI(content="let me test that", id="ai-1", tool_calls=[
        {"name": "run_code", "args": {}, "id": "c1"},
        {"name": "run_command", "args": {}, "id": "c2"}])
    n, written = _repair([HumanMessage("hi"), dangling,
                          HumanMessage("still there?"), HumanMessage("any news?")])
    assert n == 2
    assert [type(m).__name__ for m in written] == ["RemoveMessage"]
    assert written[0].id == "ai-1"


def test_repair_ignores_tool_calls_that_were_answered():
    """A completed turn deeper in the history must not be mistaken for a dangling one."""
    from langchain_core.messages import AIMessage as AI, HumanMessage, ToolMessage as TM
    n, written = _repair([
        HumanMessage("hi"),
        AI(content="", id="ai-1", tool_calls=[{"name": "t", "args": {}, "id": "c1"}]),
        TM(content="done", tool_call_id="c1"),
        AI(content="answer"), HumanMessage("more")])
    assert (n, written) == (0, [])


def test_repair_is_a_no_op_on_a_completed_turn():
    from langchain_core.messages import AIMessage as AI, HumanMessage
    n, appended = _repair([HumanMessage("hi"), AI(content="done")])
    assert (n, appended) == (0, [])


def test_a_buried_dangling_call_is_removed_not_answered_in_the_wrong_place():
    """This assertion used to say the opposite — that a buried call is somebody else's bug
    and must be left alone. A live thread then wedged permanently: LangGraph writes the new
    HumanMessage before calling the model, so one killed turn buries its own dangling call
    and every later question buries it deeper. Appending is still wrong (a toolResult must
    come *immediately* after its toolUse), so the repair removes the message instead."""
    from langchain_core.messages import AIMessage as AI, HumanMessage, RemoveMessage
    buried = AI(content="", id="ai-old", tool_calls=[{"name": "t", "args": {}, "id": "old"}])
    n, written = _repair([buried, HumanMessage("hi"), HumanMessage("still there?")])
    assert n == 1
    assert [type(m).__name__ for m in written] == ["RemoveMessage"]
    assert written[0].id == "ai-old"


def test_a_second_downstream_system_is_absent_not_broken():
    """Google unconfigured must cost nothing: no session build failure, no tool, and /auth
    google says so instead of pretending. The same convention as search and long-term
    memory — an optional system that is not deployed simply is not there."""
    with mock.patch.dict(agent_core._DOWNSTREAMS, {}, clear=True):
        r = agent_core.reauth("lark:ou_x", "google")
    assert r["needs_auth"] is False
    assert "未配置" in r["reply"]


def test_reauth_still_refuses_an_idp_nobody_wired_up():
    """The list is explicit on purpose: a typo should not silently start a consent flow for
    a provider that does not exist."""
    r = agent_core.reauth("lark:ou_x", "dropbox")
    assert r["needs_auth"] is False and "尚未接入" in r["reply"]


def test_the_generic_vault_fetch_does_not_claim_larks_ownership_check():
    """Only the Lark path can ask the vendor whose token this is. A generic fetch cannot, and
    saying otherwise would be the more dangerous mistake — so the docstring states the limit
    and the second system's tool reports the account it reached."""
    import inspect
    src = inspect.getsource(agent_core.lark_3lo.get_user_token_for)
    assert "_belongs_to" not in src
    assert "cannot make the same" in src or "cannot" in src


def test_the_thread_comes_from_the_router_never_from_a_client():
    """A thread is addressed by user alone, so a claimed id would read somebody else's
    conversation. The router owns rotation (/reset, /new) and is the only caller trusted to
    name one; a surface that cannot — the browser — inherits what it last named, which works
    because both share the router's runtime session id and land in one container."""
    agent_core._thread_named.clear()
    derived = agent_core._session_id_for("lark:ou_x")
    assert agent_core.thread_for("lark:ou_x") == derived      # before anything is named
    rotated = derived + "-0123456789ab"
    agent_core.remember_thread("lark:ou_x", rotated)
    assert agent_core.thread_for("lark:ou_x") == rotated
    # An empty value must not erase what the router said, nor may another user's thread.
    agent_core.remember_thread("lark:ou_x", "")
    agent_core.remember_thread("lark:ou_x", agent_core._session_id_for("lark:ou_other"))
    assert agent_core.thread_for("lark:ou_x") == rotated
    # And it is per actor.
    assert agent_core.thread_for("lark:ou_other") == agent_core._session_id_for("lark:ou_other")


def test_the_page_names_its_thread_only_alongside_a_verified_token():
    """It names the thread the router handed it, so /reset on the page takes effect; that is
    safe only because server.py refuses a thread outside the token's actor."""
    import pathlib as _p, server, inspect
    page = (_p.Path(__file__).parents[1] / "webui" / "index.html").read_text()
    assert "memorySessionId: session.memorySessionId" in page
    assert "runtimeSessionId" in page      # the routing key, which is not a capability
    assert "thread_belongs(" in inspect.getsource(server._bind_caller)


def test_a_message_arriving_mid_turn_is_steered_not_refused():
    """Two turns on one thread each read the other's half-written checkpoint, which Bedrock
    rejects — the failure that made a live thread look broken. The second message joins the
    running turn instead, so nothing has to be re-sent."""
    agent_core._pending_steer.clear()
    agent_core.steer("t-1", "and also check the logs")
    agent_core.steer("t-1", "one more thing")
    mw = [m for m in agent_core._middleware() if getattr(m, "name", "") == "steering"][0]
    # The runtime's real shape on LangGraph 1.2: the thread id lives on execution_info,
    # and there is no `config` attribute at all. A fake carrying `config` kept this test
    # green while steering was dead in production — every queued message was dropped.
    runtime = type("R", (), {"execution_info": type("E", (), {"thread_id": "t-1"})()})()
    assert not hasattr(runtime, "config")
    out = mw.before_model({"messages": []}, runtime)
    assert [m.content for m in out["messages"]] == ["and also check the logs", "one more thing"]
    # Drained, so the next model call in the same turn does not see them twice.
    assert mw.before_model({"messages": []}, runtime) is None


def test_steering_only_injects_before_a_model_call():
    """A toolResult must follow its toolUse immediately, so injecting anywhere else would
    recreate the very error this fixes. before_model is the one safe boundary: tool results
    from the previous step are written, the next model call has not happened."""
    mw = [m for m in agent_core._middleware() if getattr(m, "name", "") == "steering"][0]
    assert hasattr(mw, "before_model")
    # Nothing may hook the tool boundary, where the adjacency rule would be broken.
    assert not hasattr(agent_core._SteeringMiddleware, "after_model")
    assert not hasattr(agent_core._SteeringMiddleware, "wrap_tool_call")


def test_steering_is_per_thread():
    """One user's aside must not land in another user's turn."""
    agent_core._pending_steer.clear()
    agent_core.steer("t-a", "for A")
    mw = [m for m in agent_core._middleware() if getattr(m, "name", "") == "steering"][0]
    assert mw.before_model({"messages": []}, _runtime("t-b")) is None
    assert [q[:3] for q in agent_core._pending_steer["t-a"]] == [("for A", "", False)]
    assert list(agent_core._pending_steer) == ["t-a"]


def _load_busy_helpers():
    """The real functions, with the counter reset so tests don't inherit each other."""
    agent_core._in_flight = 0
    return {"busy": agent_core.busy, "_track": agent_core._track}


def test_busy_reflects_in_flight_turns():
    ns = _load_busy_helpers()
    busy, track = ns["busy"], ns["_track"]
    assert busy() is False
    track(+1)
    assert busy() is True
    track(+1)
    track(-1)
    assert busy() is True          # still one turn running
    track(-1)
    assert busy() is False         # idle again, so the container may be reclaimed


def test_track_is_usable_from_a_thread():
    """The decrement happens on the worker thread; a scoping bug shows up there."""
    import threading
    ns = _load_busy_helpers()
    busy, track = ns["busy"], ns["_track"]
    track(+1)
    err = []

    def worker():
        try:
            track(-1)
        except Exception as e:  # noqa: BLE001
            err.append(e)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert not err, f"_track failed off the main thread: {err}"
    assert busy() is False


# ------------------------------- web search ---------------------------------
# Search is optional; with no gateway configured the agent must simply run
# without the tool rather than failing to build a session.

def test_websearch_gateway_contract():
    """The protocol-version header: without it the Gateway negotiates 2025-03-26 and
    rejects with -32022."""
    src = open(os.path.join(os.path.dirname(__file__), "websearch.py"), encoding="utf-8").read()
    assert "MCP-Protocol-Version" in src
    assert "2025-11-25" in src
    # available() must be a pure config check, so an unset gateway just means no tool
    assert 'os.environ.get("WEBSEARCH_GATEWAY_URL", "")' in src


# --------------------------- deferred authorization -------------------------
# A user without a vaulted Lark token still gets a working session: lark-mcp lists
# its tools without one and only rejects tools/call. Consent is raised when a tool
# is actually reached, so plain chat ("hello") is never gated on it.

def test_auth_marker_matches_the_mcp_server():
    """The client detects the auth wall by matching lark-mcp's rejection text. It is
    a cross-process string contract: if the server's wording changes and this does
    not, the user silently stops being offered a consent link."""
    here = os.path.dirname(__file__)
    core = open(os.path.join(here, "agent_core.py"), encoding="utf-8").read()
    marker = re.search(r'_NEEDS_TOKEN_MARKER = "([^"]+)"', core).group(1)
    for name in ("lark-cli", "approval", "google"):
        server = open(os.path.join(here, "..", "mcp-servers", name, "server.js"), encoding="utf-8").read()
        assert marker in server, (
            f"agent_core expects {marker!r} but mcp-servers/{name}/server.js no longer says it")


def test_hit_auth_wall_reads_tool_results_not_the_final_reply():
    """The check must inspect tool results, not the model's text answer: the model
    paraphrases errors, so 'no user token is available' in the reply slips past a string
    check on the reply — verified end-to-end before this fix."""
    hit = agent_core._hit_auth_wall_from_tool_results
    lark = {"consent": {"lark": "https://consent"}, "tool_idp": {"lark_docs": "lark"}}

    # 1. Nothing collected → nothing to inspect, so no auth wall.
    assert hit(lark) is None
    assert hit({**lark, "tool_texts": []}) is None
    # 2. A paraphrase of the error, not the tool's own words → NOT a wall. This is
    #    exactly the case that fooled the first version of this function.
    assert hit({**lark, "tool_texts": [
        ("lark_docs", "It looks like no user token is available — please authorize")]}) is None
    # 3. The tool result carrying the marker verbatim → wall, for that tool's system.
    assert hit({**lark, "tool_texts": [("lark_docs", "no user token (authorize first)")]}) == "lark"
    # 4. Same tool result but the system is granted → no wall, because then the refusal
    #    means something else and consent won't fix it.
    assert hit({"tool_idp": {"lark_docs": "lark"}, "consent": {},
                "tool_texts": [("lark_docs", "no user token (authorize first)")]}) is None


def test_unauthorized_session_passes_an_empty_token_and_still_gets_tools():
    """The token is passed as an empty header rather than skipping the connection —
    verified against the deployed Runtime, which returns the full tool list for an
    empty token. Skipping it would leave the model unaware of its Lark tools."""
    opened = []

    async def fake_open(stack, connection):
        opened.append(connection["headers"])
        return [_dummy_tool]

    with mock.patch.object(agent_core.lark_3lo, "get_user_lark_token",
                           return_value=("auth_url", "https://consent")), \
         mock.patch.object(agent_core.lark_3lo, "mcp_connection_for",
                           side_effect=lambda tok, url="": {"headers": {"tok": tok}}), \
         mock.patch.object(agent_core, "_aopen_tools", fake_open), \
         mock.patch.object(agent_core.websearch, "available", return_value=False):
        s = agent_core._build_session("lark:ou_x", "", "mem1")

    assert s["auth_url"] == "https://consent"      # consent is remembered, not raised
    assert s["consent"] == {"lark": "https://consent"}
    assert s["tool_idp"] == {_dummy_tool.name: "lark"}
    assert opened == [{"tok": ""}], "the server must still be connected, with no token"
    assert s["graph"] is not None, "the model must see the Lark tools regardless"


def test_a_web_users_lark_is_granted_on_demand_like_google():
    """A web user has no Lark identity to match, so the Lark-user path — which mints a link at
    build and refuses a grant whose open_id differs (wrong_owner) — would lock them out for
    good. For them Lark is one more system: read without minting, its tools listed, and a
    link minted only when they ask to authorise."""
    lark_user_path, opened = [], []

    async def fake_open(stack, connection):
        opened.append(connection.get("url"))
        return [_dummy_tool]

    with mock.patch.object(agent_core.lark_3lo, "get_user_lark_token",
                           side_effect=lambda *a, **k: lark_user_path.append(a)), \
         mock.patch.object(agent_core.lark_3lo, "peek_user_token_for", return_value=""), \
         mock.patch.object(agent_core.lark_3lo, "mcp_connection_for",
                           side_effect=lambda tok, url="": {"url": url or "lark-mcp"}), \
         mock.patch.object(agent_core, "_aopen_tools", fake_open), \
         mock.patch.object(agent_core.websearch, "available", return_value=False):
        s = agent_core._build_session("web:sub-carol", "",
                                      agent_core._session_id_for("web:sub-carol"))
        assert lark_user_path == []
        assert "lark-mcp" in opened and s["consent"]["lark"] == ""
        assert not s["auth_url"] and not s["identity_error"]
        assert agent_core.auth_status("web:sub-carol")["lark"] is False
        with mock.patch.object(agent_core.lark_3lo, "get_user_token_for",
                               return_value=("auth_url", "https://lark-consent")) as generic:
            r = agent_core.reauth("web:sub-carol", "lark")
    assert r["auth_url"] == "https://lark-consent" and generic.call_count == 1
    assert lark_user_path == []


# ----------------------------- streaming to a card --------------------------
# chat_async types the answer into a CardKit streaming card so the user isn't left
# staring at silence for the ~7.5 s before the first token. Two things must hold:
# updates are throttled (not one call per token), and any CardKit failure falls back
# to send_text so the answer is never lost.

from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel


@tool
def _dummy_tool() -> str:
    """A tool the scripted model can call. Returns lark-mcp's refusal text."""
    return "no user token (authorize first)"


class _ScriptedModel(BaseChatModel):
    """Streams the turns it is handed. Content arrives as a list of typed blocks, the
    way Bedrock Converse sends it, so the block-flattening path is exercised too."""

    turns: list = []
    calls: int = 0

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    async def _astream(self, messages, stop=None, run_manager=None, **kw):
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        self.__dict__["calls"] = self.calls + 1
        for text in turn.get("text", []):
            yield ChatGenerationChunk(message=AIMessageChunk(
                content=[{"type": "text", "text": text, "index": 0}]))
        for name in turn.get("tool_calls", []):
            yield ChatGenerationChunk(message=AIMessageChunk(
                content=[], tool_call_chunks=[
                    {"name": name, "args": "{}", "id": "c1", "index": 0}]))

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(message=AIMessageChunk(content=""))])


_thread_seq = iter(range(1, 10_000))


def _make_tool(name):
    """A no-op tool under an arbitrary name, so the scripted model can call the real
    thing (lark_list_my_docs, WebSearch___WebSearch) rather than a name the graph has
    never heard of."""
    from langchain_core.tools import StructuredTool
    return StructuredTool.from_function(
        func=lambda: "no user token (authorize first)", name=name,
        description=f"test double for {name}")


def _session_for(deltas, tool_calls=None, follow_up="", **extra):
    """A session dict backed by a real compiled graph. `tool_calls` are requested after
    the deltas, so the abort path runs exactly where it does in production."""
    turns = [{"text": list(deltas), "tool_calls": list(tool_calls or [])}]
    if tool_calls:
        turns.append({"text": [follow_up] if follow_up else []})
    tools = [_dummy_tool] + [_make_tool(n) for n in (tool_calls or [])]
    graph = agent_core.create_agent(model=_ScriptedModel(turns=turns), tools=tools)
    s = {"graph": graph, "tool_texts": [],
         "config": {"configurable": {"thread_id": f"t{next(_thread_seq)}"}}}
    s.update(extra)
    return s


def _patch_notify(card, notify_sent):
    return mock.patch.object(
        agent_core, "lark_notify",
        mock.Mock(StreamingCard=lambda chat_id, **kw: card,
                  send_text=lambda cid, t: notify_sent.append(t) or True))


class FakeCard:
    def __init__(self, open_ok=True, update_ok=True, close_ok=True):
        self._open_ok, self._update_ok, self._close_ok = open_ok, update_ok, close_ok
        self.ok = False
        self.updates = []
        self.closed_with = None
        self.tool_events = []

    def open(self):
        self.ok = self._open_ok
        return self._open_ok

    def update(self, full_text):
        if not self.ok:
            return False
        self.updates.append(full_text)
        self.ok = self._update_ok
        return self._update_ok

    def close(self, final_text, handed_over=False):
        self.closed_with = final_text
        self.handed_over = handed_over
        return self._close_ok

    def tool_started(self):
        self.tool_events.append("start")

    def tool_finished(self):
        self.tool_events.append("finish")

    def compacting(self, started):
        self.tool_events.append("compact" if started else "compacted")



# ------------------- vaulted token ownership (consent hijack) ------------------

def _load_3lo_guard(owner_lookup):
    """Exec the ownership check in isolation, so the HTTP lookup can be replaced without
    touching the module's boto3 client at import time."""
    src = open(os.path.join(os.path.dirname(__file__), "lark_3lo.py"), encoding="utf-8").read()
    start = src.index("_VERIFIED: dict[str, str] = {}")
    end = src.index("def get_user_lark_token(")
    ns = {"hashlib": hashlib, "log": mock.Mock(), "httpx": mock.Mock()}
    exec(src[start:end], ns)  # nosec B102 # nosemgrep: exec-detected — asserts a source contract
    ns["_token_owner"] = owner_lookup          # replace the HTTP call
    return ns


def test_vaulted_token_must_belong_to_the_actor():
    """Consent completion binds the token to whatever userId the return-url was told
    (`state`), not to the account that actually signed in — so forwarding a consent
    link vaults someone else's token under your name. Checked at the point of use."""
    ns = _load_3lo_guard(lambda t: "ou_alice")
    assert ns["_belongs_to"]("tok", "lark:ou_alice") is True
    assert ns["_belongs_to"]("tok2", "lark:ou_bob") is False     # Bob's grant, Alice's slot


def test_a_wrong_owner_is_refused_without_burning_the_real_grant():
    """The vault is keyed by the caller's workload identity, not by the actor named in the
    request, so a fetch for a bogus actor returns the caller's own token. Forcing a fresh
    flow on that mismatch destroyed the caller's working grant — measured: a token at
    13:16:40, gone one second later, and every later fetch answered authorizationUrl."""
    import inspect
    src = inspect.getsource(agent_core.lark_3lo.get_user_lark_token)
    branch = src[src.index("if token:"):src.index('return "wrong_owner"')]
    assert "force=True" not in branch      # forcing here is what burned the grant
    # Forcing when the vault is *empty* is a different case and is required — see the
    # completability test below.
    assert "force=True" in src[src.index('return "wrong_owner"'):]


def test_an_empty_vault_mints_a_completable_consent_link():
    """An authorizationUrl from forceAuthentication=False cannot be completed —
    CompleteResourceTokenAuth answers "Invalid or expired session" for it, repeatably, while
    the same flow forced completes. So the empty case must re-ask with force."""
    import inspect
    src = inspect.getsource(agent_core.lark_3lo.get_user_lark_token)
    tail = src[src.index('return "wrong_owner"'):]
    assert "if not force:" in tail and "force=True" in tail


def test_the_session_retry_hands_back_the_live_consent_link():
    """Each build mints a pending consent session and invalidates the previous one, so
    keeping the first attempt's auth_url gave the user a link the retry had already killed —
    CompleteResourceTokenAuth then answered "Invalid or expired session" every time."""
    calls = []

    def build(actor, email, mem_sid, wat="", own_claim=False):
        calls.append(1)
        return {"auth_url": f"https://consent/{len(calls)}", "mem_sid": mem_sid,
                "created": 0, "stack": None}

    with mock.patch.object(agent_core, "_build_session", side_effect=build), \
         mock.patch.object(agent_core, "_close_session"):
        s = agent_core._get_session("lark:ou_x", "", "mem1", workload_token="wat")
    assert len(calls) == 2
    assert s["auth_url"] == "https://consent/2"   # the newest, not the first


def test_unverifiable_token_owner_fails_closed():
    """An owner we cannot establish is not "probably fine"."""
    ns = _load_3lo_guard(lambda t: "")
    assert ns["_belongs_to"]("tok", "lark:ou_alice") is False


def test_verified_token_is_not_rechecked_every_turn():
    """One whoami per token, not per turn — the check is on the hot path."""
    calls = []
    def owner(t):
        calls.append(t)
        return "ou_alice"
    ns = _load_3lo_guard(owner)
    for _ in range(4):
        assert ns["_belongs_to"]("tok", "lark:ou_alice") is True
    assert len(calls) == 1


def test_the_ownership_cache_is_bounded():
    """A long-lived container sees many users; the cache must not grow without limit."""
    ns = _load_3lo_guard(lambda t: t.replace("tok-", "ou_"))
    for i in range(ns["_VERIFIED_MAX"] + 20):
        ns["_belongs_to"](f"tok-{i}", f"lark:ou_{i}")
    assert len(ns["_VERIFIED"]) <= ns["_VERIFIED_MAX"]



# ------------------------ CardKit worker (coalescing) ------------------------

def _load_card(monkeypatch_calls):
    """Import StreamingCard with its HTTP layer replaced. lark_notify imports boto3
    only, so unlike agent_core it loads on the test host."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import importlib
    import lark_notify
    importlib.reload(lark_notify)
    lark_notify._tenant_token = lambda: "tok"

    def fake_call(method, url, body, bearer=""):
        content = body.get("content")
        if content is None and "elements" in body:       # a block's first write
            content = json.loads(body["elements"])[0].get("content")
        # Only the answer's text is under test here; status writes are counted elsewhere.
        if "/elements/p" in url or "/elements/tail" in url or url.endswith("/settings"):
            content = None
        monkeypatch_calls.append((method, content, body.get("sequence")))
        return {"code": 0}
    lark_notify._call = fake_call
    return lark_notify


def test_card_worker_writes_the_newest_text_and_never_loses_the_last():
    """Superseded states may be dropped — CardKit takes the full text every time, so
    the newest write subsumes the earlier ones. The FINAL text must not be dropped,
    which is why close() stops the worker and writes synchronously."""
    calls = []
    ln = _load_card(calls)
    card = ln.StreamingCard("oc_1")
    card.card_id = "c1"
    card.ok = True
    import threading
    card._worker = threading.Thread(target=card._pump, daemon=True)
    card._worker.start()

    for t in ("a", "ab", "abc", "abcd"):
        assert card.update(t) is True
    card.close("abcd-final")

    contents = [c for _, c, _ in calls if c]
    assert "abcd-final" in contents, "the final text must always be written"
    assert contents[-2:][0] == "abcd-final" or contents[-1] == "abcd-final"
    # Every write is one of the accumulated states, never a stale fragment reordered.
    assert all(c in ("a", "ab", "abc", "abcd", "abcd-final") for c in contents)


def test_card_sequence_never_repeats_or_goes_backwards():
    """CardKit rejects an out-of-order sequence, and the worker plus close() both
    write — so the counter has to be shared and monotonic across them."""
    calls = []
    ln = _load_card(calls)
    card = ln.StreamingCard("oc_1")
    card.card_id = "c1"
    card.ok = True
    import threading
    card._worker = threading.Thread(target=card._pump, daemon=True)
    card._worker.start()
    for i in range(6):
        card.update("x" * (i + 1))
    card.close("done")
    seqs = [s for _, _, s in calls if s is not None]
    assert seqs == sorted(set(seqs)), f"sequence not strictly increasing: {seqs}"


def _recording_card():
    """A StreamingCard whose writes are recorded as (method, path, body)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import importlib
    import lark_notify
    importlib.reload(lark_notify)
    lark_notify._tenant_token = lambda: "tok"
    writes = []

    def fake_call(method, url, body, bearer=""):
        writes.append((method, url.split("/cards/c1", 1)[-1], body))
        return {"code": 0}
    lark_notify._call = fake_call
    card = lark_notify.StreamingCard("oc_1")
    card.card_id, card.ok, card.sent = "c1", True, True
    return lark_notify, card, writes


def _inserted(writes):
    """(element_id, after) for every element added to the card, in order."""
    return [(json.loads(b["elements"])[0]["element_id"], b["target_element_id"])
            for m, p, b in writes if m == "POST" and p == "/elements"]


def _shown(writes, eid):
    """The last thing CardKit was told a status line says, inserted or replaced."""
    def label(el):
        inner = el["columns"][0]["elements"][0]
        if inner["tag"] == "column_set":            # live: label and dots side by side
            inner = inner["columns"][0]["elements"][0]
        return inner["content"]
    last = None
    for m, p, b in writes:
        if m == "POST" and p == "/elements":
            el = json.loads(b["elements"])[0]
            if el["element_id"] == eid:
                last = label(el)
        elif m == "PUT" and p == f"/elements/{eid}":
            last = label(json.loads(b["element"]))
    return last


def _deleted(writes):
    return [p.split("/")[-1] for m, p, _ in writes if m == "DELETE"]


def test_status_lines_sit_between_the_texts_where_they_happened():
    """Text, a tool, more text: the tool's line goes after the first text and stays there
    as a record, and the second text goes after it. Nothing already shown moves."""
    ln, card, writes = _recording_card()
    card.update("我先把仓库拉下来。"); card._flush()
    card.tool_started(); card._flush()
    card.tool_finished()
    card.update("我先把仓库拉下来。下载成功，看一下结构："); card._flush()
    assert card.close("我先把仓库拉下来。下载成功，看一下结构：") is True
    assert _inserted(writes) == [("t1", "p0"), ("p2", "t1"), ("t3", "p2"), ("tail", "t3")]
    # The opening "thinking" line was only a wait, so it goes; the tool line stays.
    assert _deleted(writes) == ["p0"]
    assert "🔧 调用工具 · " in _shown(writes, "p2") and "思考中" not in _shown(writes, "p2")
    t3 = [json.loads(b["elements"])[0]["content"] for m, p, b in writes
          if m == "POST" and "t3" in b["elements"]]
    assert t3 == ["下载成功，看一下结构："], "each block holds only its own text"
    assert writes[-1][1] == "/settings"          # streaming turned off last


def test_consecutive_tool_calls_share_one_line():
    ln, card, writes = _recording_card()
    card.update("先查一下。"); card._flush()
    for _ in range(2):
        card.tool_started(); card.tool_finished()
    card.update("先查一下。查到了。")
    card.close("先查一下。查到了。")
    pills = [i for i, _ in _inserted(writes) if i.startswith("p")]
    assert pills == ["p2"]
    assert "调用工具 ×2" in _shown(writes, "p2") and "思考中" not in _shown(writes, "p2")


def test_the_tool_count_comes_from_results_not_from_starts():
    """A streamed tool call names its tool on more than one chunk, so counting starts
    would overcount; a result is one call that really ran."""
    ln, card, _ = _recording_card()
    card.tool_started(); card.tool_started(); card.tool_finished()
    assert "1 次工具调用" in card.summary()


def test_a_turn_without_tools_says_only_how_long_it_took():
    ln, card, _ = _recording_card()
    line = card.summary()
    assert line.startswith("⏱️ 耗时") and "工具" not in line


def test_the_summary_is_a_full_width_band_at_the_end():
    ln, card, writes = _recording_card()
    card.update("答案")
    assert card.close("答案") is True
    tail = [json.loads(b["elements"])[0] for m, p, b in writes
            if m == "POST" and '"tail"' in b["elements"]]
    assert tail and tail[0]["background_style"] == "grey-50"
    assert tail[0]["columns"][0]["width"] == "weighted"
    assert "⏱️ 耗时" in tail[0]["columns"][0]["elements"][0]["content"]


def test_a_handed_over_card_says_so_and_the_next_one_replies_and_keeps_the_clock():
    ln, card, writes = _recording_card()
    card.tool_started(); card.tool_finished()
    card.update("先看一下。")
    assert card.close("先看一下。", handed_over=True) is True
    tail = [json.loads(b["elements"])[0] for m, p, b in writes
            if m == "POST" and '"tail"' in b["elements"]][0]
    assert "被用户新消息打断" in tail["columns"][0]["elements"][0]["content"]
    nxt = ln.StreamingCard("oc_1", reply_to="om_steered", carry=card)
    assert nxt._t0 == card._t0 and nxt._tools_done == 1 and nxt._steps == card._steps
    posted = []
    ln._post = lambda url, body, bearer="": posted.append(url) or {
        "code": 0, "data": {"card_id": "c2"}}
    ln._animation_keys = lambda: {}
    assert nxt.open() is True
    nxt._stop.set(); nxt._wake.set()
    assert posted[-1].endswith("/open-apis/im/v1/messages/om_steered/reply")


def test_a_failed_status_line_does_not_cost_the_answer():
    """A status line is decoration. If CardKit refuses one the answer is still delivered
    and the card closed, so the caller must not fall back to posting text again."""
    ln, card, writes = _recording_card()
    orig = ln._call
    ln._call = lambda m, u, b, bearer="": (
        {"code": 1, "msg": "no"} if m == "POST" and '"column_set"' in b.get("elements", "")
        else orig(m, u, b, bearer))
    card.update("先查。"); card._flush()
    card.tool_started(); card.tool_finished()
    card.update("先查。好了。")
    assert card.close("先查。好了。") is True
    assert ("t3", "t1") in _inserted(writes), "text anchors on the text before it"


def test_a_consent_prompt_replaces_the_half_sentence():
    ln, card, writes = _recording_card()
    card.update("让我查一下"); card._flush()
    card.close("🔐 这一步需要访问你的 Lark 账号")
    assert [b["content"] for m, p, b in writes if p == "/elements/t1/content"] == [
        "🔐 这一步需要访问你的 Lark 账号"]


def test_past_the_component_limit_the_answer_keeps_writing_without_new_lines():
    """Measured: a card holds 200 components, and an insert past that is refused."""
    ln, card, writes = _recording_card()
    ln._MAX_GAPS = 1
    text = ""
    for part in ("一。", "二。", "三。"):
        text += part
        card.update(text); card._flush()
        card.tool_started(); card.tool_finished(); card._flush()
    card.close(text)
    ids = [i for i, _ in _inserted(writes)]
    assert ids == ["t1", "p2", "t3", "tail"]
    assert [b["content"] for m, p, b in writes if p == "/elements/t3/content"][-1] == "二。三。"


def test_a_wait_writes_only_when_its_state_changes():
    """The live line moves by itself, so a long tool call costs no writes while it runs."""
    import threading, time as _t
    ln, card, writes = _recording_card()
    card._keys = {"think": "img_think", "work": "img_work"}
    card._worker = threading.Thread(target=card._pump, daemon=True)
    card._worker.start()
    card.tool_started()
    _t.sleep(1.5)  # nosemgrep: arbitrary-sleep — a clock would have written by now
    card._stop.set(); card._wake.set(); card._worker.join(timeout=2)
    replaced = [json.loads(b["element"]) for m, p, b in writes if p == "/elements/p0"]
    assert len(replaced) == 1, "one write for thinking → tool, none while it runs"
    dots = replaced[0]["columns"][0]["elements"][0]["columns"][1]["elements"][0]
    assert dots["img_key"] == "img_work"
    assert _shown(writes, "p0") == "<font color='grey'>🔧 调用工具</font>"


def test_thinking_leaves_a_line_with_how_long_it_took():
    ln, card, writes = _recording_card()
    card._phase_since -= 3                       # three seconds of thought
    card.update("答案"); card.close("答案")
    assert _shown(writes, "p0") == "<font color='grey'>🤔 思考 · 3 秒</font>"
    assert "p0" not in _deleted(writes)


def test_the_status_animations_ship_with_the_image():
    """Read from agent/assets at run time; the Dockerfile has to copy them, and the image
    tag has to change with them or a new animation is never built."""
    import pathlib
    sys.path.insert(0, os.path.dirname(__file__))
    import lark_notify
    here = pathlib.Path(__file__).parent
    for name in lark_notify._ANIM_FILES.values():
        assert (here / "assets" / name).read_bytes()[:6] in (b"GIF89a", b"GIF87a"), name
    assert "COPY assets/" in (here / "Dockerfile").read_text()
    assert 'glob.glob("assets/*")' in (here.parent / "scripts" / "provision.sh").read_text()


def test_a_failed_animation_upload_is_not_retried_on_every_turn():
    sys.path.insert(0, os.path.dirname(__file__))
    import importlib
    import lark_notify
    importlib.reload(lark_notify)
    with mock.patch.object(lark_notify, "upload_image",
                           side_effect=lark_notify.LarkError("no")) as up:
        assert lark_notify._animation_keys() == {}
        assert lark_notify._animation_keys() == {}
    assert up.call_count == 1


def test_card_update_reports_failure_on_the_next_call():
    """A write now fails on the worker, so the caller learns one call late. That is
    enough to stop streaming and fall back — the answer is never lost."""
    calls = []
    ln = _load_card(calls)
    ln._call = lambda *a, **k: {"code": 99991400, "msg": "nope"}
    card = ln.StreamingCard("oc_1")
    card.card_id = "c1"
    card.ok = True
    import threading, time as _t
    card._worker = threading.Thread(target=card._pump, daemon=True)
    card._worker.start()
    assert card.update("a") is True        # queued before any failure is known
    for _ in range(50):                   # let the worker discover it
        if not card.ok:
            break
        _t.sleep(0.02)  # nosemgrep: arbitrary-sleep — polls the worker in a unit test
    assert card.ok is False
    assert card.update("ab") is False      # now the caller is told


def test_a_failed_card_stops_writing_instead_of_racing_the_text_fallback():
    """Once the caller has posted the answer as text, a worker still retrying the card
    could land the same answer there too when CardKit recovers."""
    import threading, time as _t
    ln, card, writes = _recording_card()
    ln._call = lambda *a, **k: {"code": 99991400, "msg": "nope"}
    card._worker = threading.Thread(target=card._pump, daemon=True)
    card._worker.start()
    card.update("a")
    for _ in range(50):
        if not card.ok:
            break
        _t.sleep(0.02)  # nosemgrep: arbitrary-sleep — polls the worker in a unit test
    assert card.close("a") is False
    assert not card._worker.is_alive()


def test_a_given_up_card_does_not_keep_its_dots():
    """The answer goes out as text, so nothing on the card may still look busy — not the
    live line, and not one already marked for removal when the failure stopped the flush."""
    ln, card, writes = _recording_card()
    orig = ln._call
    ln._call = lambda m, u, b, bearer="": (
        {"code": 1, "msg": "no"} if m == "POST" and '"t1"' in b.get("elements", "")
        else orig(m, u, b, bearer))
    card.update("答案"); card._flush()           # p0 is dead, t1 is refused
    assert card.ok is False
    assert card.close("答案") is False
    assert "p0" in _deleted(writes)
    assert writes[-1][1] == "/settings"


def test_a_network_error_is_a_failed_write_not_an_exception():
    """_call only knew HTTPError; a timeout or an HTML 502 escaped close() and skipped the
    text fallback."""
    sys.path.insert(0, os.path.dirname(__file__))
    import importlib
    import lark_notify
    importlib.reload(lark_notify)           # earlier tests replace _call itself
    with mock.patch.object(lark_notify.urllib.request, "urlopen",
                           side_effect=TimeoutError("slow")):
        assert lark_notify._call("PUT", "https://x", {})["code"] == -1


def test_stream_hands_over_every_delta_cumulatively():
    """The loop no longer throttles: handing over text is a lock, not a round trip, so
    every delta goes over and the card's worker decides what to actually write. Each
    hand-over carries the full text so far — CardKit renders the appended tail, so a
    non-prefix would flash the wrong content."""
    card = FakeCard()
    deltas = ["a" * 30, "b" * 30, "c" * 30, "d" * 30]
    session = _session_for(deltas)
    with _patch_notify(card, []):
        text = agent_core._stream_to_chat(session, "msg", "oc_1")
    assert text == "a"*30 + "b"*30 + "c"*30 + "d"*30
    assert len(card.updates) == len(deltas)
    for u in card.updates:
        assert text.startswith(u)          # cumulative, always a prefix of the whole
    assert card.closed_with == text


def test_stream_aborts_when_an_unauthorized_session_reaches_a_lark_tool():
    """The model must not get to narrate a refusal at length before the consent card.
    Aborting at the tool call is what keeps the card the only thing the user reads."""
    card = FakeCard()
    session = _session_for(["让我查一下…"], tool_calls=["approval_list_pending"],
                           follow_up="很抱歉，我没有权限……",
                           consent={"lark": "https://consent"},
                           tool_idp={"approval_list_pending": "lark"})
    with _patch_notify(card, []):
        text = agent_core._stream_to_chat(session, "查待审批", "oc_1")
    assert session.get("walled_tool") == "approval_list_pending"
    # The half-sentence is replaced, not left on the card, and the model's apology for
    # the refusal never runs at all.
    assert "让我查一下" not in text
    assert "很抱歉" not in text
    assert "Lark" in text


def test_stream_does_not_abort_on_websearch_when_unauthorized():
    """Search needs no Lark grant, so an unauthorized user must still get results."""
    card = FakeCard()
    session = _session_for(["天气是…"], tool_calls=["WebSearch___WebSearch"],
                           follow_up="晴天", consent={"lark": "https://consent"})
    with _patch_notify(card, []):
        text = agent_core._stream_to_chat(session, "今天天气", "oc_1")
    assert "walled_tool" not in session
    assert text == "天气是…晴天"           # the turn ran to completion


def test_the_card_hears_of_a_tool_only_after_the_words_before_it():
    """The graph runs on the agent loop and the card is fed from the consumer thread. A
    tool event sent straight from the loop could overtake the last words before it, and
    the card would cut the text block there — the tool line lost, a fragment left alone.
    So tool events travel in the same queue as the text."""
    class Timeline(FakeCard):
        def __init__(self):
            super().__init__()
            self.seen = []

        def tool_started(self):
            self.seen.append(("start", self.updates[-1] if self.updates else ""))

        def tool_finished(self):
            self.seen.append(("finish", self.updates[-1] if self.updates else ""))

    card = Timeline()
    session = _session_for(["我先", "看一下。"], tool_calls=["lark_lookup"],
                           follow_up="查到了。")
    with _patch_notify(card, []):
        agent_core._stream_to_chat(session, "msg", "oc_1")
    assert card.seen[0] == ("start", "我先看一下。")
    assert ("finish", "我先看一下。") in card.seen
    assert card.updates[-1] == "我先看一下。查到了。"


def test_a_message_steered_in_mid_turn_is_answered_on_a_new_card_below_it():
    """Answering it on the card above the message read as the agent ignoring it. The card
    in progress is closed where the message joined, and the rest goes on a new card —
    quoting the message in a group, and carrying the turn's clock so the last card sums
    up the whole turn."""
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.tools import StructuredTool
    thread = f"t{next(_thread_seq)}"

    def lookup() -> str:
        """Steers a message in while the turn is between model calls, as a user would."""
        agent_core.steer(thread, "不用看代码了")
        agent_core._steer_reply_to[thread] = "om_steered"
        return "found"
    steering = type("SteeringMiddleware", (AgentMiddleware,),
                    dict(agent_core._SteeringMiddleware.__dict__))()
    graph = agent_core.create_agent(
        model=_ScriptedModel(turns=[{"text": ["我先查一下。"], "tool_calls": ["lookup"]},
                                    {"text": ["好的，不看了。"]}]),
        tools=[_dummy_tool, StructuredTool.from_function(func=lookup, name="lookup")],
        middleware=[steering])
    session = {"graph": graph, "tool_texts": [],
               "config": {"configurable": {"thread_id": thread}}}
    made = []

    def factory(chat_id, reply_to="", carry=None):
        card = FakeCard()
        card.reply_to, card.carry = reply_to, carry
        made.append(card)
        return card
    notify = mock.Mock(StreamingCard=factory, send_text=lambda cid, t: True,
                       is_group=lambda cid: True)
    try:
        with mock.patch.object(agent_core, "lark_notify", notify):
            text = agent_core._stream_to_chat(session, "看一下仓库", "oc_group")
    finally:
        agent_core._pending_steer.pop(thread, None)
    first, second = made
    assert (first.closed_with, first.handed_over) == ("我先查一下。", True)
    assert second.reply_to == "om_steered" and second.carry is first
    assert (second.closed_with, second.handed_over) == ("好的，不看了。", False)
    assert second.updates == ["好的，不看了。"], "the new card holds only what came after"
    assert text == "我先查一下。好的，不看了。"


def test_a_card_that_showed_nothing_yet_is_not_handed_over():
    """Steered in before any text — B sent while A's session was still being built — the
    card already sits where the answer belongs, and closing it would leave it empty."""
    from langchain.agents.middleware import AgentMiddleware
    thread = f"t{next(_thread_seq)}"
    agent_core.steer(thread, "B")
    agent_core._steer_reply_to[thread] = "om_b"
    steering = type("SteeringMiddleware", (AgentMiddleware,),
                    dict(agent_core._SteeringMiddleware.__dict__))()
    graph = agent_core.create_agent(model=_ScriptedModel(turns=[{"text": ["A 和 B 的答案"]}]),
                                    tools=[_dummy_tool], middleware=[steering])
    session = {"graph": graph, "tool_texts": [],
               "config": {"configurable": {"thread_id": thread}}}
    made = []
    notify = mock.Mock(StreamingCard=lambda chat_id, **kw: made.append(FakeCard()) or made[-1],
                       send_text=lambda cid, t: True)
    with mock.patch.object(agent_core, "lark_notify", notify):
        agent_core._stream_to_chat(session, "A", "oc_1")
    assert len(made) == 1 and made[0].closed_with == "A 和 B 的答案"


def test_a_failed_hand_over_still_delivers_the_rest():
    """The hand-over runs inside the stream: if it raises, the turn must go on as text
    rather than end in an apology with the graph still running."""
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.tools import StructuredTool
    thread = f"t{next(_thread_seq)}"

    def lookup() -> str:
        """Steers a message in mid-turn."""
        agent_core.steer(thread, "还有这个")
        return "found"
    steering = type("SteeringMiddleware", (AgentMiddleware,),
                    dict(agent_core._SteeringMiddleware.__dict__))()
    graph = agent_core.create_agent(
        model=_ScriptedModel(turns=[{"text": ["先查。"], "tool_calls": ["lookup"]},
                                    {"text": ["好了。"]}]),
        tools=[_dummy_tool, StructuredTool.from_function(func=lookup, name="lookup")],
        middleware=[steering])
    session = {"graph": graph, "tool_texts": [],
               "config": {"configurable": {"thread_id": thread}}}
    first, sent = FakeCard(), []

    def factory(chat_id, **kw):
        if kw.get("carry") is not None:
            raise RuntimeError("tenant token refresh failed")
        return first
    notify = mock.Mock(StreamingCard=factory, send_text=lambda cid, t: sent.append(t) or True,
                       is_group=lambda cid: False)
    try:
        with mock.patch.object(agent_core, "lark_notify", notify):
            agent_core._stream_to_chat(session, "看一下", "oc_1")
    finally:
        agent_core._pending_steer.pop(thread, None)
    assert first.closed_with == "先查。" and sent == ["好了。"]


def test_stream_does_not_abort_when_authorized():
    """An authorized session has no auth_url, so tool calls proceed normally."""
    card = FakeCard()
    session = _session_for(["结果…"], tool_calls=["lark_list_my_docs"],
                           follow_up="共 3 个文件")     # no auth_url
    with _patch_notify(card, []):
        text = agent_core._stream_to_chat(session, "查文档", "oc_1")
    assert "walled_tool" not in session
    assert text == "结果…共 3 个文件"


def test_stream_collects_tool_results_for_the_auth_wall_fallback():
    """A turn that runs to completion still has to expose what the tools said, or the
    fallback auth-wall check has nothing to read."""
    card = FakeCard()
    # No auth_url while streaming, so the abort (which normally fires first) stays out
    # of the way and the turn runs to completion — the fallback's actual situation is a
    # session that looked authorized until a tool refused mid-turn.
    session = _session_for(["查询中…"], tool_calls=["lark_list_my_docs"],
                           follow_up="失败了", tool_idp={"lark_list_my_docs": "lark"})
    with _patch_notify(card, []):
        agent_core._stream_to_chat(session, "查文档", "oc_1")
    session["consent"] = {"lark": "https://consent"}   # the vault lookup that follows finds none
    assert any(agent_core._NEEDS_TOKEN_MARKER in t for _, t in session["tool_texts"])
    assert agent_core._hit_auth_wall(session) == "lark"


def test_stream_falls_back_to_text_when_card_cannot_open():
    """No cardkit:card:write scope → open() fails → the answer still arrives as text."""
    card = FakeCard(open_ok=False)
    sent = []
    session = _session_for(["hello ", "world"])
    with _patch_notify(card, sent):
        text = agent_core._stream_to_chat(session, "msg", "oc_1")
    assert text == "hello world"
    assert card.updates == []              # never tried to stream
    assert sent == ["hello world"]         # delivered as plain text instead


def test_stream_falls_back_when_an_update_fails_midway():
    """A card that dies mid-stream must still deliver the full answer via text."""
    card = FakeCard(update_ok=False)       # first update flips ok to False
    sent = []
    session = _session_for(["x" * 100, "y" * 100])
    with _patch_notify(card, sent):
        text = agent_core._stream_to_chat(session, "msg", "oc_1")
    assert text == "x"*100 + "y"*100
    assert sent == [text]                  # fell back after the failed update
    assert card.closed_with is None        # never reached a clean close


def test_fresh_session_evicts_a_cached_unauthorized_session():
    """The consent-resume replay must not reuse the cached unauthorized session: its
    MCP sessions hold an empty token, so the turn would wall again. Measured in the
    field — consent completed 31 s after the prompt, inside _UNAUTH_TTL, so waiting
    for expiry is not a fix."""
    closed, built = [], []

    def fake_build(actor_id, email, mem_sid, workload_token="", own_claim=False):
        built.append(mem_sid)
        return {"created": 1e9}                      # authorized: no auth_url

    cached = {"auth_url": "https://consent", "created": 1e9, "stack": object()}
    with mock.patch.dict(agent_core._sessions, {"lark:u|mem1": cached}, clear=True), \
         mock.patch.object(agent_core, "_build_session", fake_build), \
         mock.patch.object(agent_core, "_close_session", lambda s: closed.append(s)), \
         mock.patch.object(agent_core.time, "time", return_value=1e9 + 10):
        # Without fresh, the cached unauthorized session is returned (still inside TTL).
        s = agent_core._get_session("lark:u", "", "mem1")
        assert s.get("auth_url") == "https://consent"
        assert built == []

        # With fresh, it is evicted, its MCP sessions closed, and a new one built.
        s = agent_core._get_session("lark:u", "", "mem1", fresh=True)

    assert closed == [cached], "the stale MCP sessions must be closed, not leaked"
    assert built == ["mem1"]
    assert "auth_url" not in s


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# --------------------------- thread bookkeeping ------------------------------
# /status and /clear moved here from the router when the conversation became LangGraph
# state. Both must answer without building a session — that would re-handshake every MCP
# server for what is a bookkeeping question.

def _fake_saver(messages=None, deleted=None):
    class _S:
        def __init__(self):
            self.deleted = deleted

        async def aget_tuple(self, config):
            if messages is None:
                return None
            return type("T", (), {"checkpoint": {"channel_values": {"messages": messages}}})()

        async def adelete_thread(self, thread_id):
            self.deleted = thread_id
    return _S()


def test_history_stats_counts_a_turn_only_when_an_answer_was_delivered():
    """A tool-calling AIMessage is work in progress, not a turn — it is counted as a call."""
    from langchain_core.messages import AIMessage as AI, ToolMessage as TM
    msgs = [HumanMessage("q"), AI(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
            TM(content="r", tool_call_id="1"), AI(content="a")]
    with mock.patch.object(agent_core, "_checkpointer", return_value=_fake_saver(msgs)):
        assert agent_core.history_stats("lark:ou_x", "sess-1") == {
            "turns": 1, "toolCalls": 1, "messages": 3,
            "contextTokens": 0, "cachedTokens": 0,
            "modelCalls": 0, "inputTotal": 0, "cachedTotal": 0}


def test_history_stats_reports_zero_for_an_untouched_thread():
    with mock.patch.object(agent_core, "_checkpointer", return_value=_fake_saver(None)):
        assert agent_core.history_stats("lark:ou_x", "sess-1") == {
            "turns": 0, "toolCalls": 0, "messages": 0,
            "contextTokens": 0, "cachedTokens": 0,
            "modelCalls": 0, "inputTotal": 0, "cachedTotal": 0}


def test_context_size_is_the_latest_calls_billed_input_not_an_estimate():
    """From the usage Bedrock reported, shaped as a real checkpoint stores it: input_tokens
    already includes what came from the cache. An estimate would not do — the library's
    counter charges a flat 85 tokens per image."""
    from langchain_core.messages import AIMessage as AI
    old = AI(content="a", usage_metadata={"input_tokens": 4968, "output_tokens": 49,
             "total_tokens": 5017, "input_token_details": {"cache_read": 4832}})
    new = AI(content="b", usage_metadata={"input_tokens": 9147, "output_tokens": 517,
             "total_tokens": 9664, "input_token_details": {"cache_read": 5049}})
    msgs = [HumanMessage("q"), old, HumanMessage("q2"), new]
    with mock.patch.object(agent_core, "_checkpointer", return_value=_fake_saver(msgs)):
        out = agent_core.history_stats("lark:ou_x", "sess-1")
    assert (out["contextTokens"], out["cachedTokens"]) == (9147, 5049)
    # Summed over every call still in the checkpoint, for the thread's hit rate.
    assert (out["modelCalls"], out["inputTotal"], out["cachedTotal"]) == (2, 14115, 9881)


def _compactable(turns: int, summary: str):
    """A real agent graph with `turns` finished turns, and the summary model replaced."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langgraph.checkpoint.memory import InMemorySaver
    graph = agent_core.create_agent(model=_ScriptedModel(turns=[{"text": ["ans"]}]),
                                    tools=[], checkpointer=InMemorySaver(),
                                    middleware=agent_core._middleware()[:2])
    config = {"configurable": {"thread_id": f"c{next(_thread_seq)}"}}
    for i in range(turns):
        agent_core._run(graph.ainvoke({"messages": [HumanMessage(f"q{i}")]}, config=config))
    session = {"graph": graph, "config": config}
    return (session, mock.patch.object(agent_core, "_get_session", return_value=session),
            mock.patch.object(agent_core, "_summary_model",
                              return_value=FakeListChatModel(responses=[summary])))


def test_compact_replaces_the_older_turns_with_a_summary_and_the_thread_goes_on():
    session, got, model = _compactable(turns=6, summary="Earlier: q0 to q2 were asked.")
    with got, model, mock.patch.object(agent_core, "_COMPACT_KEEP", 4):
        out = agent_core.compact("lark:ou_x", "t")
    msgs = agent_core._run(session["graph"].aget_state(session["config"])).values["messages"]
    assert out["compacted"] > 0 and out["kept"] == sum(1 for m in msgs if not agent_core._is_summary(m))
    assert agent_core._is_summary(msgs[0]) and "q0 to q2" in agent_core._text_of(msgs[0])
    # Nothing pending: the next turn starts from the summary, as on a live thread.
    assert agent_core._run(session["graph"].aget_state(session["config"])).next == ()
    agent_core._run(session["graph"].ainvoke({"messages": [HumanMessage("next")]},
                                              config=session["config"]))


def test_compact_refuses_a_fallback_summary_rather_than_erase_the_history():
    """The failure the threshold path was fixed for — a history replaced by the fallback
    line — must not come back through the manual door."""
    session, got, model = _compactable(turns=6, summary=agent_core._SUMMARY_FALLBACK)
    before = agent_core._run(session["graph"].aget_state(session["config"])).values["messages"]
    with got, model, mock.patch.object(agent_core, "_COMPACT_KEEP", 4):
        out = agent_core.compact("lark:ou_x", "t")
    after = agent_core._run(session["graph"].aget_state(session["config"])).values["messages"]
    assert out == {"compacted": 0, "reason": "failed"}
    assert [m.id for m in after] == [m.id for m in before]


def test_compact_leaves_a_short_thread_and_a_running_turn_alone():
    session, got, model = _compactable(turns=1, summary="s")
    with got, model, mock.patch.object(agent_core, "_COMPACT_KEEP", 4):
        assert agent_core.compact("lark:ou_x", "t")["reason"] == "short"
    with mock.patch.object(agent_core, "thread_busy", return_value=True):
        assert agent_core.compact("lark:ou_x", "t") == {"compacted": 0, "reason": "busy"}


def test_the_tool_overview_keeps_assembly_order_and_names_the_missing_grants():
    session = {"tool_source": {"lark_whoami": "lark", "approval_list": "approval",
                               "run_code": "code"},
               "tool_idp": {"lark_whoami": "lark", "approval_list": "lark"},
               "consent": {"google": ""}}
    out = agent_core.tools_overview(session)
    assert [t["name"] for t in out["tools"]] == ["lark_whoami", "approval_list", "run_code"]
    assert out["tools"][1] == {"name": "approval_list", "source": "approval", "idp": "lark"}
    assert out["ungranted"] == ["google"]
    assert agent_core.tools_overview({"identity_error": "x"}) == {
        "tools": [], "identityError": "x"}


def test_transcript_rebuilds_what_a_reloaded_page_should_show():
    from langchain_core.messages import AIMessage as AI, ToolMessage as TM
    msgs = [HumanMessage("我有几份文档"),
            AI(content="", tool_calls=[{"name": "list_docs", "args": {"n": 5}, "id": "c1"}]),
            TM(content="5 份", tool_call_id="c1"),
            AI(content="你有 5 份。")]
    with mock.patch.object(agent_core, "_checkpointer", return_value=_fake_saver(msgs)):
        out = agent_core.transcript("lark:ou_x", "sess-1")
    assert [m["role"] for m in out["messages"]] == [
        "user", "toolCall", "toolResult", "assistant"]
    # The id is what pairs a call with its result when the page rebuilds the card.
    assert out["messages"][1]["id"] == out["messages"][2]["id"] == "c1"
    assert '"n": 5' in out["messages"][1]["args"]
    assert out["truncated"] is False


def test_transcript_reads_multimodal_content_without_crashing():
    """A turn that carried an image has a list of blocks, not a string."""
    with mock.patch.object(agent_core, "_checkpointer", return_value=_fake_saver(
            [HumanMessage(content=[{"type": "text", "text": "看图"},
                                   {"type": "image", "source": {}}])])):
        out = agent_core.transcript("lark:ou_x", "sess-1")
    assert out["messages"] == [{"role": "user", "content": "看图"}]


def test_transcript_caps_a_long_thread_and_says_so():
    msgs = [HumanMessage(f"q{i}") for i in range(agent_core._TRANSCRIPT_LIMIT + 10)]
    with mock.patch.object(agent_core, "_checkpointer", return_value=_fake_saver(msgs)):
        out = agent_core.transcript("lark:ou_x", "sess-1")
    assert len(out["messages"]) == agent_core._TRANSCRIPT_LIMIT
    assert out["truncated"] is True
    assert out["messages"][-1]["content"] == f"q{len(msgs) - 1}"   # the tail, not the head


def test_transcript_is_blank_rather_than_broken_when_unreadable():
    broken = mock.Mock()
    broken.aget_tuple.side_effect = RuntimeError("no table")
    with mock.patch.object(agent_core, "_checkpointer", return_value=broken):
        out = agent_core.transcript("lark:ou_x", "sess-1")
    assert out == {"messages": [], "unavailable": True}


def test_the_transcript_reads_the_verified_callers_thread():
    import inspect
    import server
    src = inspect.getsource(server.handle_invocations)
    i = src.index('action == "transcript"')
    block = src[i:i + 500]
    assert 'payload["actorId"]' in block and "agent_core.thread_for(actor_id)" in block


def test_history_stats_degrades_instead_of_failing_status():
    """/status must still render if the checkpoint cannot be read."""
    broken = mock.Mock()
    broken.aget_tuple.side_effect = RuntimeError("no table")
    with mock.patch.object(agent_core, "_checkpointer", return_value=broken):
        assert agent_core.history_stats("lark:ou_x", "sess-1")["unavailable"] is True


def test_clear_history_deletes_the_thread_and_drops_the_cached_session():
    """A surviving cached session holds a compiled graph whose next turn would write on
    top of a thread the user was told is gone."""
    saver = _fake_saver()
    closed = []
    agent_core._sessions["lark:ou_x|sess-1"] = {"stack": None, "created": 0}
    with mock.patch.object(agent_core, "_checkpointer", return_value=saver), \
         mock.patch.object(agent_core, "_close_session", closed.append):
        assert agent_core.clear_history("lark:ou_x", "sess-1") == {"deleted": True}
    assert saver.deleted == "sess-1"
    assert "lark:ou_x|sess-1" not in agent_core._sessions
    assert len(closed) == 1


def test_clear_history_reports_failure_rather_than_claiming_success():
    broken = mock.Mock()
    broken.adelete_thread.side_effect = RuntimeError("boom")
    with mock.patch.object(agent_core, "_checkpointer", return_value=broken):
        r = agent_core.clear_history("lark:ou_x", "sess-1")
    assert r["deleted"] is False and r["error"] == "RuntimeError"


# --------------------------- prompt caching ----------------------------------
# Bedrock caches nothing without a cachePoint block, and the only route that works is a
# per-call kwarg: model_kwargs is rerouted to additional_model_request_fields, and
# .bind(cache_control=...) is dropped when create_agent calls bind_tools. Both measured —
# so these tests exist to catch the day a library change breaks the injection silently.

class _CaptureClient:
    def __init__(self):
        self.params = {}

    def converse(self, **kw):
        self.params = kw
        return {"output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}


def _invoke_capturing(**ctor):
    from langchain_core.messages import SystemMessage
    client = _CaptureClient()
    m = agent_core._CachingChatBedrockConverse(
        model="global.anthropic.claude-sonnet-5", region_name="us-west-2",
        client=client, **ctor)
    m.bind_tools([{"name": "t", "description": "d",
                   "input_schema": {"type": "object", "properties": {}}}]).invoke(
        [SystemMessage("sys"), HumanMessage("a"), HumanMessage("b")])
    p = client.params
    return {
        "system": sum(1 for b in p.get("system", []) if "cachePoint" in b),
        "tools": sum(1 for t in p.get("toolConfig", {}).get("tools", []) if "cachePoint" in t),
        "messages": sum(1 for msg in p.get("messages", [])
                        for b in (msg.get("content") or [])
                        if isinstance(b, dict) and "cachePoint" in b),
        "ttls": [b["cachePoint"].get("ttl") for b in p.get("system", []) if "cachePoint" in b],
    }


def test_cache_points_reach_the_request_on_system_tools_and_messages():
    got = _invoke_capturing(cache_ttl="1h")
    assert got["system"] == 1 and got["tools"] == 1 and got["messages"] >= 1
    assert got["ttls"] == ["1h"]


def test_five_minute_ttl_is_sent_without_an_explicit_ttl_field():
    """Bedrock's default window is 5m, and langchain-aws omits the field for it rather
    than sending a redundant value — so an absent ttl here is correct, not a bug."""
    assert _invoke_capturing(cache_ttl="5m")["ttls"] == [None]


def test_empty_ttl_disables_caching_entirely():
    """An escape hatch that must really disable it: a cachePoint still costs a 1.25-2x
    write on every request that misses."""
    got = _invoke_capturing(cache_ttl="")
    assert (got["system"], got["tools"], got["messages"]) == (0, 0, 0)


@pytest.mark.parametrize("model,explicit", [
    ("global.anthropic.claude-sonnet-5", True),
    ("us.anthropic.claude-haiku-4-5-20251001-v1:0", True),
    ("amazon.nova-pro-v1:0", True),
    ("global.moonshotai.kimi-k3", False),
    ("zai.glm-5", False),
    ("deepseek.v3.2", False),
])
def test_only_claude_and_nova_get_a_cache_point(model, explicit):
    """Bedrock's explicit-caching table for Converse lists only those two families; Kimi K3
    rejects any request with a cachePoint (measured), and caches implicitly instead."""
    assert agent_core._caches_explicitly(model) is explicit


@pytest.mark.parametrize("model,setting,ttl,mode", [
    ("global.anthropic.claude-sonnet-5", "1h", "1h", "explicit 1h"),
    ("global.anthropic.claude-sonnet-5", "", "5m", "explicit 5m"),
    ("global.anthropic.claude-sonnet-5", "off", "", "off"),
    ("global.moonshotai.kimi-k3", "1h", "", "implicit"),
    ("global.moonshotai.kimi-k3", "off", "", "implicit"),
])
def test_the_model_decides_whether_a_ttl_is_sent(model, setting, ttl, mode):
    import importlib
    with mock.patch.dict(os.environ, {"BEDROCK_MODEL_ID": model, "PROMPT_CACHE_TTL": setting}):
        m = importlib.reload(agent_core)
        assert (m._CACHE_TTL, m.cache_mode(), m._model.cache_ttl) == (ttl, mode, ttl)
    importlib.reload(agent_core)


def test_a_mistyped_cache_setting_stops_the_container():
    import importlib
    with mock.patch.dict(os.environ, {"PROMPT_CACHE_TTL": "1hr"}), pytest.raises(SystemExit):
        importlib.reload(agent_core)
    importlib.reload(agent_core)


def test_an_explicit_per_call_cache_control_is_not_overridden():
    from langchain_core.messages import SystemMessage
    client = _CaptureClient()
    m = agent_core._CachingChatBedrockConverse(
        model="global.anthropic.claude-sonnet-5", region_name="us-west-2",
        client=client, cache_ttl="1h")
    m.invoke([SystemMessage("sys"), HumanMessage("a")], cache_control={"ttl": "5m"})
    assert [b["cachePoint"].get("ttl") for b in client.params["system"]
            if "cachePoint" in b] == [None]      # 5m → field omitted


# --------------------------- long-term memory --------------------------------
# Two tools, not automatic extraction. The isolation that matters is the namespace: it is
# built from the bound actor, never from a model-supplied argument, so no prompt can make
# recall read another user's records.

def test_no_memory_resource_means_no_tools_rather_than_an_error():
    import memory_tools
    with mock.patch.object(memory_tools, "_MEMORY_ID", ""):
        assert memory_tools.tools_for("lark:ou_x") == []


def test_the_namespace_comes_from_the_bound_actor_not_from_arguments():
    import memory_tools
    calls = {}

    class _C:
        def batch_create_memory_records(self, **kw):
            calls["write"] = kw
            return {"successfulRecords": [{"memoryRecordId": "r1"}]}

        def retrieve_memory_records(self, **kw):
            calls["read"] = kw
            return {"memoryRecordSummaries": [{"content": {"text": "code name is X"}}]}

    with mock.patch.object(memory_tools, "_MEMORY_ID", "mem-1"), \
         mock.patch.object(memory_tools.boto3, "client", return_value=_C()):
        remember, recall = memory_tools.tools_for("lark:ou_alice")
        assert remember.invoke({"fact": "likes tea"}) == "Saved."
        assert "code name is X" in recall.invoke({"query": "code name?"})

    assert calls["write"]["records"][0]["namespaces"] == ["/facts/lark:ou_alice"]
    assert calls["read"]["namespace"] == ["/facts/lark:ou_alice"][0]
    # No strategy id: measured that the service embeds and scores directly-written records
    # without one, so declaring a strategy would only add cost and constraints.
    assert "memoryStrategyId" not in calls["write"]["records"][0]
    assert set(remember.args) == {"fact"} and set(recall.args) == {"query"}


def test_a_rejected_record_is_reported_not_swallowed():
    import memory_tools

    class _C:
        def batch_create_memory_records(self, **kw):
            return {"successfulRecords": [], "failedRecords": [{"errorCode": "Throttled"}]}

    with mock.patch.object(memory_tools, "_MEMORY_ID", "mem-1"), \
         mock.patch.object(memory_tools.boto3, "client", return_value=_C()):
        remember, _ = memory_tools.tools_for("lark:ou_x")
        assert "could not save" in remember.invoke({"fact": "x"})


def test_recall_says_so_when_there_is_nothing_stored():
    """An empty result must not read as a tool failure, or the model retries it."""
    import memory_tools

    class _C:
        def retrieve_memory_records(self, **kw):
            return {"memoryRecordSummaries": []}

    with mock.patch.object(memory_tools, "_MEMORY_ID", "mem-1"), \
         mock.patch.object(memory_tools.boto3, "client", return_value=_C()):
        _, recall = memory_tools.tools_for("lark:ou_x")
        assert recall.invoke({"query": "anything?"}) == "Nothing on record about that."


# --------------------------- bounding the thread -----------------------------
# The thread is permanent per user and DynamoDBSaver has no prune, so summarisation is the
# only bound. It must trigger on a threshold, never per turn: it rewrites the prefix, and
# rewriting the prefix every turn would make the prompt cache miss every turn.

def test_summarisation_triggers_on_a_token_threshold_not_every_turn():
    mw = [m for m in agent_core._middleware()
          if type(m).__name__ == "SummarizationMiddleware"]
    assert len(mw) == 1
    clauses = mw[0]._trigger_clauses
    assert clauses == [{"tokens": agent_core._SUMMARIZE_AT_TOKENS}]
    assert agent_core._SUMMARIZE_AT_TOKENS > 0


def test_the_summariser_does_not_pay_for_prompt_caching():
    """The summary call happens once per crossing and is never re-read, so a cachePoint
    would only buy a 1.25-2x write premium."""
    summariser = [m for m in agent_core._middleware()
                  if type(m).__name__ == "SummarizationMiddleware"][0]
    assert summariser.model.cache_ttl == ""


def test_a_message_steered_in_or_answered_later_keeps_the_time_it_arrived():
    """A message steered in mid-turn, or answered as a continuation after the turn, was said
    when it arrived — not when the turn began, and not when its answer started."""
    from langgraph.checkpoint.memory import InMemorySaver
    agent_core._pending_steer.clear()
    with mock.patch.object(agent_core, "_SUMMARIZE_AT_TOKENS", 0):
        mw = agent_core._middleware()
    tool = _make_tool("lookup")
    graph = agent_core.create_agent(
        model=_ScriptedModel(turns=[{"text": ["looking"], "tool_calls": ["lookup"]},
                                    {"text": ["done"]}, {"text": ["later"]}]),
        tools=[tool], middleware=mw, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-steer-clock", "received_ms": 1000}}
    agent_core.steer("t-steer-clock", "also this", "h-steer")
    arrived = agent_core._pending_steer["t-steer-clock"][0][3]

    async def run(inputs):
        async for _ in graph.astream({"messages": inputs}, cfg, stream_mode="messages"):
            pass
    agent_core._run(run([HumanMessage("q", id="h-q")]))
    agent_core._run(run(agent_core._turn_input([("and later", "h-late", False, 5000)])))
    by_id = {m.id: m.additional_kwargs.get("ts")
             for m in agent_core._run(graph.aget_state(cfg)).values["messages"]}
    assert by_id["h-q"] == 1000 and by_id["h-steer"] == arrived and by_id["h-late"] == 5000


def test_a_message_whose_turn_stopped_before_a_reply_still_gets_its_time():
    """Stopped at a consent wall the model never returns, so the clock never stamps it."""
    from langgraph.checkpoint.memory import InMemorySaver
    with mock.patch.object(agent_core, "_SUMMARIZE_AT_TOKENS", 0):
        mw = agent_core._middleware()
    graph = agent_core.create_agent(
        model=_ScriptedModel(turns=[{"text": ["one"]}, {"text": ["walled"]}, {"text": ["two"]}]),
        tools=[_dummy_tool], middleware=mw, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-stop", "received_ms": 1}}

    async def full(text, mid):
        async for _ in graph.astream({"messages": [HumanMessage(text, id=mid)]}, cfg,
                                     stream_mode="messages"):
            pass

    async def stopped(text, mid):     # abandoned at the first token, as _walls does
        stream = graph.astream({"messages": [HumanMessage(text, id=mid)]}, cfg,
                               stream_mode="messages")
        async for _ in stream:
            break
        await stream.aclose()
    agent_core._run(full("q1", "h1"))
    agent_core._run(stopped("q2", "h2"))
    assert agent_core._run(agent_core.astamp_unanswered(graph, cfg, 4242)) == 1
    assert agent_core._run(agent_core.astamp_unanswered(graph, cfg, 9999)) == 0   # idempotent
    agent_core._run(full("q3", "h3"))
    msgs = agent_core._run(graph.aget_state(cfg)).values["messages"]
    by_id = {m.id: m.additional_kwargs.get("ts") for m in msgs if isinstance(m, HumanMessage)}
    assert by_id["h2"] == 4242 and [m.id for m in msgs if isinstance(m, HumanMessage)] == ["h1", "h2", "h3"]


def test_auth_status_names_the_account_each_grant_came_from():
    """Read with the grant itself, so a user can tell which of their accounts it is; a lookup
    that fails costs the name, never the status."""
    tokens = {"lark-prov": "lark-tok", "google-prov": "g-tok"}
    with mock.patch.object(agent_core.lark_3lo, "configured", return_value=True), \
         mock.patch.object(agent_core.lark_3lo, "lark_provider", return_value=("lark-prov", [])), \
         mock.patch.object(agent_core, "_DOWNSTREAMS", {"google": ("u", "google-prov", [])}), \
         mock.patch.object(agent_core.lark_3lo, "peek_user_token_for",
                           side_effect=lambda actor, prov, scopes, wt: tokens.get(prov, "")), \
         mock.patch.dict(agent_core._ACCOUNT_OF, {"lark": lambda t: "XUESONG · me@myners.net",
                                                  "google": lambda t: "me@myners.net"}):
        st = agent_core.auth_status("web:s-1")
        assert st["lark"] is True and st["google"] is True
        assert st["accounts"] == {"lark": "XUESONG · me@myners.net", "google": "me@myners.net"}
        tokens.pop("google-prov")
        with mock.patch.dict(agent_core._ACCOUNT_OF, {"lark": lambda t: ""}):
            st = agent_core.auth_status("web:s-1")
        assert st["lark"] is True and st["google"] is False and st["accounts"] == {}


def test_reauthorising_a_system_already_granted_starts_a_new_consent():
    """How a user switches accounts: /auth google answered "Already authorized." while a grant
    was vaulted, so there was no way to consent as anybody else."""
    fetched = []

    def fake_fetch(wat, actor, force=False, provider="", scopes=None):
        fetched.append(force)
        return {"authorizationUrl": "https://new-consent"} if force else {"accessToken": "old"}
    with mock.patch.dict(agent_core._DOWNSTREAMS, {"google": ("https://g", "p", ["openid"])}), \
         mock.patch.object(agent_core.lark_3lo, "_workload_token_for", return_value="wat"), \
         mock.patch.object(agent_core.lark_3lo, "_fetch_vaulted", side_effect=fake_fetch):
        r = agent_core.reauth("web:s-1", "google")
        assert r["auth_url"] == "https://new-consent" and fetched == [True]
        # An ordinary fetch still hands back the vaulted grant.
        assert agent_core.lark_3lo.get_user_token_for("web:s-1", "p", ["openid"]) == ("token", "old")


def test_summarisation_can_be_disabled_outright():
    """Steering and image attachment stay either way — one keeps a mid-turn message from
    being lost, the other is what lets the model see a screenshot at all."""
    with mock.patch.object(agent_core, "_SUMMARIZE_AT_TOKENS", 0):
        assert [type(m).__name__ for m in agent_core._middleware()] == [
            "ImageMiddleware", "SteeringMiddleware", "ClockMiddleware"]


def test_each_message_records_when_it_was_said():
    """A reloaded page shows the times Lark chat shows, so they live on the thread itself:
    the user's message and the reply each carry one, and the transcript hands them out."""
    from langgraph.checkpoint.memory import InMemorySaver
    with mock.patch.object(agent_core, "_SUMMARIZE_AT_TOKENS", 0):
        mw = agent_core._middleware()
    graph = agent_core.create_agent(
        model=_ScriptedModel(turns=[{"text": ["one"]}, {"text": ["two"]}]), tools=[_dummy_tool],
        middleware=mw, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-clock"}}
    before = int(time.time() * 1000)
    # The user's message carries when the turn arrived, not when the model was first called.
    arrived = {"configurable": {**cfg["configurable"], "received_ms": before - 5000}}
    async def turn(text, mid):
        async for _ in graph.astream({"messages": [HumanMessage(text, id=mid)]}, cfg,
                                     stream_mode="messages"):
            pass
    async def first(text, mid):
        async for _ in graph.astream({"messages": [HumanMessage(text, id=mid)]}, arrived,
                                     stream_mode="messages"):
            pass
    agent_core._run(first("q1", "h1"))
    agent_core._run(turn("q2", "h2"))
    after = int(time.time() * 1000)
    msgs = agent_core._run(graph.aget_state(cfg)).values["messages"]
    assert [type(m).__name__ for m in msgs] == ["HumanMessage", "AIMessage"] * 2
    stamps = [m.additional_kwargs.get("ts") for m in msgs]
    assert stamps[0] == before - 5000
    assert all(before <= t <= after for t in stamps[1:]) and stamps == sorted(stamps), stamps
    # Stamped in place: still one copy of each.
    assert [agent_core._text_of(m) for m in msgs] == ["q1", "one", "q2", "two"]
    with mock.patch.object(agent_core, "_checkpointer", return_value=graph.checkpointer), \
         mock.patch.object(agent_core, "_thread_config", return_value=cfg):
        out = agent_core.transcript("web:s-1")["messages"]
    assert [m.get("ts") for m in out] == stamps


def _runtime(thread_id):
    """The runtime's real shape on LangGraph 1.2: the thread id is on execution_info."""
    return type("R", (), {"execution_info": type("E", (), {"thread_id": thread_id})()})()


def _image_mw():
    return [m for m in agent_core._middleware() if getattr(m, "name", "") == "images"][0]


_BLOCK = {"type": "image", "base64": "Zm9v", "mime_type": "image/jpeg"}


def _register(thread_id, handles):
    """Register attachments with a turn, with the Lark fetch replaced."""
    with mock.patch.object(agent_core.images, "blocks_for", return_value=[_BLOCK]):
        return agent_core.images.add_to_turn(thread_id, handles)


def test_the_pixels_reach_the_request_but_never_the_thread():
    """The whole point: the model sees the image on this turn, and the stored thread keeps
    only a handle, so the image is not re-billed on every later turn."""
    from langchain_core.messages import HumanMessage as _HM
    marks, mid = _register("t-px", [{"kind": "image", "key": "img_y", "message_id": "om_x"}])
    stored = _HM(id=mid, content="what is in this?\n" + marks)
    request = type("R", (), {"messages": [stored], "runtime": _runtime("t-px")})()
    sent = _image_mw()._attach(request).messages[-1]
    assert [b["type"] for b in sent.content] == ["text", "image"]
    # Same id, so this replaces the message in the request rather than growing the thread.
    assert sent.id == mid
    # The stored message is untouched: still text, still carrying the handle.
    assert stored.content == "what is in this?\n[image 1 msg=om_x key=img_y]"
    agent_core.images.end_turn("t-px")


def test_a_run_with_no_images_is_left_alone():
    from langchain_core.messages import HumanMessage as _HM
    request = type("R", (), {"messages": [_HM(id="m1", content="just text")],
                             "runtime": _runtime("t-none")})()
    assert _image_mw()._attach(request).messages[-1].content == "just text"


def test_a_picture_sent_mid_turn_reaches_the_running_turn_on_its_own_message():
    """A message steered into a running turn joins a run whose config is already fixed, so
    its pixels come from the turn registry — bound to its own id. Attaching to "the last
    human message" would have handed the first picture to the steered one instead."""
    from langchain_core.messages import HumanMessage as _HM, AIMessage as _AI
    first_marks, first = _register("t-mid", [{"kind": "image", "key": "img_a", "message_id": "om_1"}])
    later_marks, later = _register("t-mid", [{"kind": "image", "key": "img_b", "message_id": "om_2"}])
    # Numbered across the turn, so the second picture is "image 2", not another "image 1".
    assert "[image 1 " in first_marks and "[image 2 " in later_marks
    request = type("R", (), {"runtime": _runtime("t-mid"), "messages": [
        _HM(id=first, content="look"), _AI(content="ok"), _HM(id=later, content="and this")]})()
    sent = _image_mw()._attach(request).messages
    assert [b["type"] for b in sent[0].content] == ["text", "image"]
    assert [b["type"] for b in sent[2].content] == ["text", "image"]
    agent_core.images.end_turn("t-mid")
    assert agent_core.images.pixels_for("t-mid") == {}


def test_an_attachment_steered_into_a_running_turn_is_not_dropped():
    """It used to be: the busy check came before the attachments were read, and steering
    carried text only, so a file sent while the bot was answering vanished silently."""
    agent_core._pending_steer.clear()
    agent_core._in_flight_threads.clear()
    agent_core._track(+1, "t-busy", "lark", "oc_1")
    try:
        with mock.patch.object(agent_core.images, "blocks_for", return_value=[]), \
             mock.patch.object(agent_core, "lark_notify"):
            # From another chat — a DM and a group share the thread — it is not steered: the
            # answer would appear in the chat the turn is answering.
            other = agent_core.chat_async("lark:ou_x", "elsewhere", "oc_2", mem_sid="t-busy")
            assert other["busy"] == "lark-other" and "t-busy" not in agent_core._pending_steer
            out = agent_core.chat_async("lark:ou_x", "also this", "oc_1", mem_sid="t-busy",
                                        image_handles=[{"kind": "file", "key": "f_1",
                                                        "name": "q3.pdf", "message_id": "om_9"}])
        assert out["steered"] is True
        (text, mid, _fresh, _arrived), = agent_core._pending_steer["t-busy"]
        assert "also this" in text and "save_attachment(1)" in text and "q3.pdf" in text
        assert agent_core.images.handles_for("t-busy")[0]["key"] == "f_1"
    finally:
        agent_core._track(-1, "t-busy")
        agent_core.images.end_turn("t-busy")
        agent_core._pending_steer.clear()


def test_a_lark_message_during_a_web_turn_is_told_not_steered():
    """Steered into a web turn, its answer would appear on the page while the user waits in
    Lark. Nothing is registered either, so the web turn's attachment numbering is untouched."""
    agent_core._pending_steer.clear()
    agent_core._in_flight_threads.clear()
    agent_core._track(+1, "t-web", "web")
    try:
        with mock.patch.object(agent_core, "lark_notify") as notify:
            out = agent_core.chat_async("lark:ou_x", "hi", "oc_1", mem_sid="t-web",
                                        image_handles=[{"kind": "file", "key": "f_1",
                                                        "message_id": "om_9"}])
        assert out["busy"] == "web"
        assert "网页" in notify.send_text.call_args.args[1]
        assert agent_core._pending_steer == {} and agent_core.images.handles_for("t-web") == []
    finally:
        agent_core._track(-1, "t-web")


def _fresh_claims():
    """Tests share the module's claim state; start each from nothing."""
    agent_core._in_flight = 0
    agent_core._in_flight_threads.clear()
    agent_core._pending_steer.clear()


class _InlineThread:
    """threading.Thread that runs its target on start(), so a turn finishes in the test."""
    def __init__(self, target=None, name=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


def test_a_claim_is_one_step_and_a_second_claimant_learns_who_holds_it():
    _fresh_claims()
    assert agent_core.try_claim("t-c", "web") is None
    assert agent_core.try_claim("t-c", "lark") == "web"
    agent_core.release("t-c")
    assert agent_core.thread_owner("t-c") is None and agent_core._in_flight == 0


def test_a_turn_releases_its_thread_and_attachments_on_every_way_out():
    """Left claimed, the thread reads as busy for good and every later message is steered
    into a turn that no longer exists."""
    handles = [{"kind": "file", "key": "f_1", "name": "a.pdf", "message_id": "om_1"}]
    for case in ("identity", "raises", "ok"):
        _fresh_claims()
        if case == "identity":
            got = mock.patch.object(agent_core, "_get_session",
                                    return_value={"identity_error": "nope"})
        elif case == "raises":
            got = mock.patch.object(agent_core, "_get_session", side_effect=RuntimeError("cold"))
        else:
            got = mock.patch.object(agent_core, "_get_session", return_value=_session_for([]))
        with got, mock.patch.object(agent_core, "lark_notify"), \
             mock.patch.object(agent_core, "_stream_to_chat", return_value="ok"), \
             mock.patch.object(agent_core.threading, "Thread", _InlineThread):
            try:
                agent_core.chat_async("lark:ou_x", "hi", "oc_1", mem_sid="t-out",
                                      image_handles=handles)
            except RuntimeError:
                assert case == "raises"
        assert agent_core.thread_owner("t-out") is None, case
        assert agent_core.images.handles_for("t-out") == [], case


def test_a_message_steered_after_the_last_model_call_is_answered_not_dropped():
    """That window is the whole streamed answer, which is when people add "also this". The
    turn keeps its claim and answers the leftover as a continuation, attachments intact."""
    _fresh_claims()
    seen = []

    def stream(session, inputs, chat_id, message_id=None, received_ms=None):
        seen.append(inputs)
        if len(seen) == 1:       # arrives while the first answer is streaming
            with mock.patch.object(agent_core.images, "blocks_for", return_value=[]):
                agent_core.chat_async("lark:ou_x", "also this", "oc_1", mem_sid="t-late",
                                      image_handles=[{"kind": "file", "key": "f_9",
                                                      "name": "late.pdf", "message_id": "om_9"}])
            assert agent_core.images.handles_for("t-late"), "still registered for the continuation"
        return "ok"

    with mock.patch.object(agent_core, "_get_session", return_value=_session_for([])), \
         mock.patch.object(agent_core, "lark_notify"), \
         mock.patch.object(agent_core, "_stream_to_chat", side_effect=stream), \
         mock.patch.object(agent_core.threading, "Thread", _InlineThread):
        agent_core.chat_async("lark:ou_x", "first", "oc_1", mem_sid="t-late")
    assert len(seen) == 2
    (text, _mid, _fresh, _arrived), = seen[1]
    assert "also this" in text and "save_attachment(1)" in text
    assert agent_core.thread_owner("t-late") is None
    assert agent_core.images.handles_for("t-late") == []


def test_across_surfaces_a_message_is_told_not_steered():
    """Steered across surfaces, the answer appears in the other window — and a sync caller
    would be promised an answer it never receives."""
    for owner, arrives in (("sync", "lark"), ("web", "lark"), ("lark", "sync"), ("web", "sync")):
        _fresh_claims()
        agent_core.try_claim("t-x", owner)
        with mock.patch.object(agent_core, "lark_notify") as notify:
            if arrives == "lark":
                out = agent_core.chat_async("lark:ou_x", "hi", "oc_1", mem_sid="t-x")
                assert out["busy"] == owner and notify.send_text.called
            else:
                out = agent_core.chat_result("lark:ou_x", "hi", mem_sid="t-x")
                assert out["busy"] == owner and "steered" not in out
        assert agent_core._pending_steer == {}, (owner, arrives)
        agent_core.release("t-x")


def test_a_web_runs_claim_lasts_until_the_run_ends_not_until_the_page_stops_reading():
    """A page closed mid-answer stops reading, but the graph runs on and writes the
    checkpoint; releasing at that point let a Lark turn start beside it."""
    import asyncio, threading as _th
    done = _th.Event()

    async def produce():
        for i in range(3):
            await asyncio.sleep(0.05)
            yield i

    async def read_one_and_leave():
        events = agent_core.stream_on_agent_loop(produce, on_done=done.set)
        async for _ in events:
            break            # the page went away after the first event
        assert not done.is_set(), "released while the run was still going"

    asyncio.run(read_one_and_leave())
    assert done.wait(5), "the run finished but never released"


def test_a_picture_steered_into_a_real_turn_reaches_the_model_on_its_own_message():
    """Through a compiled graph, not a faked request: the steering middleware injects the
    message with its id, and the image middleware finds its pixels by that id."""
    _fresh_claims()
    seen_by_model = []

    class _Recording(_ScriptedModel):
        async def _astream(self, messages, stop=None, run_manager=None, **kw):
            seen_by_model.append(messages)
            async for chunk in super()._astream(messages, stop, run_manager, **kw):
                yield chunk

    from langgraph.checkpoint.memory import InMemorySaver
    graph = agent_core.create_agent(model=_Recording(turns=[{"text": ["ok"]}]), tools=[],
                                    checkpointer=InMemorySaver(),
                                    middleware=agent_core._middleware()[:2])
    with mock.patch.object(agent_core.images, "blocks_for", return_value=[_BLOCK]):
        _, first = agent_core.images.add_to_turn(
            "t-real", [{"kind": "image", "key": "img_a", "message_id": "om_1"}])
        marks, later = agent_core.images.add_to_turn(
            "t-real", [{"kind": "image", "key": "img_b", "message_id": "om_2"}])
    agent_core.steer("t-real", "and this one\n" + marks, later)
    # Streamed, as a real turn is (_iter_deltas): that is the path that calls _astream.
    async def run():
        async for _ in graph.astream({"messages": [HumanMessage("look", id=first)]},
                                     config={"configurable": {"thread_id": "t-real"}},
                                     stream_mode="messages"):
            pass
    agent_core._run(run())
    humans = [m for m in seen_by_model[0] if isinstance(m, HumanMessage)]
    assert [m.id for m in humans] == [first, later]
    for m in humans:
        assert any(isinstance(b, dict) and b.get("type") == "image" for b in m.content), m.id
    agent_core.images.end_turn("t-real")


def test_a_malformed_page_request_does_not_leave_the_thread_claimed():
    """Claimed before the run input is built, so a payload that fails validation must
    release it — left claimed, Lark answers "正在网页上回答" for good."""
    import asyncio, agui
    _fresh_claims()
    session = {"mem_sid": "t-bad", "graph": object(), "config": {"configurable": {}}}

    async def go():
        with mock.patch.object(agent_core, "aget_session",
                               mock.AsyncMock(return_value=session)):
            await agui.handle(mock.Mock(), {"actorId": "lark:ou_x", "messages": "not a list"}, "")

    try:
        asyncio.run(go())
    except Exception:  # noqa: BLE001 — the validation error itself is not under test
        pass
    assert agent_core.thread_owner("t-bad") is None and agent_core._in_flight == 0


def test_release_is_idempotent():
    _fresh_claims()
    agent_core.try_claim("t-i", "lark")
    agent_core.release("t-i")
    agent_core.release("t-i")
    assert agent_core._in_flight == 0


def test_a_steer_queued_while_a_turn_failed_to_start_is_reported_not_carried_over():
    """Acknowledged as steered, so dropping it silently loses it; carried over, it would be
    injected into the next turn with its pixels gone and its numbers pointing elsewhere."""
    _fresh_claims()

    def cold_build(*a, **k):
        agent_core.steer("t-fail", "meanwhile", "msg-x")     # arrives during the build
        raise RuntimeError("build failed")

    with mock.patch.object(agent_core, "_get_session", side_effect=cold_build), \
         mock.patch.object(agent_core, "lark_notify") as notify:
        try:
            agent_core.chat_async("lark:ou_x", "first", "oc_1", mem_sid="t-fail")
        except RuntimeError:
            pass
    assert agent_core._pending_steer == {}
    assert any("1 条消息没来得及处理" in c.args[1] for c in notify.send_text.call_args_list)
    assert agent_core.thread_owner("t-fail") is None


def test_a_continuation_that_cannot_get_a_session_says_how_many_it_dropped():
    """Its inputs are already off the queue, so the release at the end cannot count them;
    for a consent replay the parked copy is spent too, so silence would lose it outright."""
    _fresh_claims()
    calls = {"n": 0}

    def get_session(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _session_for([])
        raise RuntimeError("mcp connect timeout")      # the continuation's rebuild

    def stream(session, inputs, chat_id, message_id=None, received_ms=None):
        if calls["n"] == 1:
            agent_core.steer("t-cont", "after consent", "m-r", fresh=True)
        return "ok"

    with mock.patch.object(agent_core, "_get_session", side_effect=get_session), \
         mock.patch.object(agent_core, "_drop_sessions"), \
         mock.patch.object(agent_core, "lark_notify") as notify, \
         mock.patch.object(agent_core, "_stream_to_chat", side_effect=stream), \
         mock.patch.object(agent_core.threading, "Thread", _InlineThread):
        agent_core.chat_async("lark:ou_x", "first", "oc_1", mem_sid="t-cont")
    sent = [c.args[1] for c in notify.send_text.call_args_list]
    assert any("1 条消息没来得及处理" in m and "RuntimeError" in m for m in sent), sent
    assert agent_core.thread_owner("t-cont") is None


def test_a_consent_replay_waits_for_the_continuation_on_a_rebuilt_session():
    """Injected into the running turn it would run on the session built before the grant
    and wall again, with the parked message already spent."""
    _fresh_claims()
    agent_core.steer("t-r", "aside", "m1")
    agent_core.steer("t-r", "replayed after consent", "m2", fresh=True)
    mw = [m for m in agent_core._middleware() if getattr(m, "name", "") == "steering"][0]
    out = mw.before_model({"messages": []}, _runtime("t-r"))
    assert [m.content for m in out["messages"]] == ["aside"]
    assert [q[:3] for q in agent_core._pending_steer["t-r"]] == [("replayed after consent", "m2", True)]
    _fresh_claims()


def test_the_claim_holder_repairs_its_own_thread():
    """A claim is taken before the session is built, so the build-time repair sees the
    thread as busy — and skipping then left a buried dangling call failing every turn."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langchain_core.messages import AIMessage as _AI
    _fresh_claims()
    graph = agent_core.create_agent(model=_ScriptedModel(turns=[{"text": ["ok"]}]), tools=[],
                                    checkpointer=InMemorySaver(),
                                    middleware=agent_core._middleware()[:2])
    cfg = {"configurable": {"thread_id": "t-rep"}}
    agent_core._run(graph.aupdate_state(cfg, {"messages": [
        HumanMessage("q"),
        _AI(content="", tool_calls=[{"name": "t", "args": {}, "id": "lost", "type": "tool_call"}]),
        HumanMessage("asked again")]}, as_node="model"))
    agent_core.try_claim("t-rep", "lark")
    try:
        assert agent_core._run(agent_core._arepair_interrupted_turn(graph, cfg)) == 0
        assert agent_core._run(agent_core._arepair_interrupted_turn(graph, cfg, own_claim=True)) == 1
    finally:
        agent_core.release("t-rep")


def test_the_card_says_when_the_thread_is_being_summarised():
    """A summary streams nothing for tens of seconds, which read as "thinking" — and looked
    stuck. The phase has its own line, its heartbeat, and its own share of the summary."""
    ln, card, _ = _recording_card()
    card.compacting(True)
    assert card._live.content == ("🗜️ 正在整理较早的对话", "work")
    # The phase lasted long enough to count: rewind its start.
    card._phase_since -= 2
    card.compacting(False)
    text, dots = card._live.content
    assert text.startswith("🗜️ 整理较早的对话 · 2 秒") and "思考中" in text and dots == "think"
    assert "整理上下文" in card.summary()


def test_a_summary_inside_a_turn_tells_that_turns_card_start_and_end():
    """Through a compiled graph with the summariser forced to fire: the card hears exactly
    when a summary is being written, and nothing when the threshold check passes."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langgraph.checkpoint.memory import InMemorySaver
    middleware = agent_core._summarization(FakeListChatModel(responses=["a summary"]),
                                           trigger=("messages", 3), keep=("messages", 1))
    graph = agent_core.create_agent(model=_ScriptedModel(turns=[{"text": ["ok"]}]), tools=[],
                                    checkpointer=InMemorySaver(), middleware=[middleware])
    cfg = {"configurable": {"thread_id": "t-sum"}}
    card = mock.Mock()
    agent_core._turn_cards["t-sum"] = card

    async def turn(text):
        async for _ in graph.astream({"messages": [HumanMessage(text)]}, config=cfg,
                                     stream_mode="messages"):
            pass
    try:
        agent_core._run(turn("one"))                 # below the trigger: no summary
        assert card.compacting.call_args_list == []
        agent_core._run(turn("two"))                 # past it: one summary
        assert [c.args for c in card.compacting.call_args_list] == [(True,), (False,)]
    finally:
        agent_core._turn_cards.pop("t-sum", None)


def test_a_broken_card_cannot_cost_the_summary():
    """The status signal runs inside the summariser; anything it raises would be read as a
    failed summary and skip the compaction."""
    agent_core._turn_cards["t-broken"] = object()          # no compacting() at all
    try:
        with mock.patch("langgraph.config.get_config",
                        return_value={"configurable": {"thread_id": "t-broken"}}):
            agent_core._turn_status("compacting", True)     # must not raise
    finally:
        agent_core._turn_cards.pop("t-broken", None)


def test_compacting_outside_a_turn_is_quiet():
    """/compact summarises outside any graph run; that is expected, not worth a traceback."""
    with mock.patch.object(agent_core.log, "warning") as warned:
        agent_core._turn_status("compacting", True)
    assert not warned.called


def test_a_turn_unregisters_its_card_even_when_it_fails():
    card = mock.Mock(open=mock.Mock(return_value=True))
    session = {"graph": object(), "config": {"configurable": {"thread_id": "t-fail-card"}},
               "tool_texts": []}
    with mock.patch.object(agent_core.lark_notify, "StreamingCard", return_value=card), \
         mock.patch.object(agent_core, "_iter_deltas", side_effect=RuntimeError("boom")):
        try:
            agent_core._stream_to_chat(session, "hi", "oc_1")
        except RuntimeError:
            pass
    assert "t-fail-card" not in agent_core._turn_cards


def test_the_sync_path_judges_consent_on_this_turns_tool_results_only():
    """The returned state is the whole thread. Scanning all of it let one old "no user
    token" result raise a consent prompt on every later sync turn."""
    from langchain_core.messages import AIMessage as _AI, ToolMessage as _TM
    seen = {}

    class _Graph:
        async def ainvoke(self, inp, config=None):
            turn = inp["messages"][0]
            seen["id"] = turn.id
            return {"messages": [
                HumanMessage("long ago", id="old"),
                _AI(content="", tool_calls=[{"name": "lark_api", "args": {}, "id": "c0"}]),
                _TM(content="no user token for this call", tool_call_id="c0", name="lark_api"),
                _AI(content="old answer"),
                turn, _AI(content="fresh answer")]}

    session = {"graph": _Graph(), "config": {"configurable": {"thread_id": "t-sync"}},
               "tool_texts": [], "consent": {}, "tool_idp": {}}
    with mock.patch.object(agent_core, "_get_session", return_value=session):
        out = agent_core.chat_result("lark:ou_x", "today's question", mem_sid="t-sync")
    assert session["tool_texts"] == [], "an old tool result leaked into this turn"
    assert out == {"reply": "fresh answer", "needs_auth": False}


def test_images_are_attached_before_anything_downstream_can_see_them():
    """First in the chain on purpose: summarisation runs on the placeholders, never on
    the bytes, and its token counter charges a flat 85 per image regardless of size."""
    assert [type(m).__name__ for m in agent_core._middleware()][0] == "ImageMiddleware"


def test_summarisation_rewrites_state_so_the_checkpoint_shrinks():
    """If it only trimmed what is sent to the model, the stored thread would keep growing
    towards DynamoDB's item cap. Asserted against the real middleware's contract."""
    from langchain_core.messages import RemoveMessage
    from langchain.agents.middleware.summarization import SummarizationMiddleware
    import inspect
    src = inspect.getsource(SummarizationMiddleware.before_model)
    assert "RemoveMessage" in src and "REMOVE_ALL_MESSAGES" in src


def test_the_summariser_sees_the_whole_slice_and_stays_off_the_chat():
    """The library trims to 4,000 tokens by default, which an agentic tail cannot fit;
    and the summary call runs inside the graph, so untagged it streams like an answer."""
    mw = [m for m in agent_core._middleware()
          if type(m).__name__ == "SummarizationMiddleware"][0]
    assert mw.trim_tokens_to_summarize is None
    assert "nostream" in mw.model.tags
    assert mw.model.metadata.get("emit-messages") is False
    assert mw.model.timeout == agent_core._SUMMARY_TIMEOUT


def test_the_turn_model_outlasts_a_silent_tool_call():
    """A model can go quiet for over a minute while it writes one large tool argument;
    the read timeout must outlast that, while a dead connection still fails to connect fast."""
    client = agent_core._model.client
    assert client.meta.config.read_timeout == agent_core._MODEL_READ_TIMEOUT >= 300
    assert client.meta.config.connect_timeout == 60


class _SummaryModel(BaseChatModel):
    """Answers every summary request with one recognisable line, streamed or not."""

    text: str = "SUMMARY-OF-EARLIER-WORK"
    calls: int = 0

    @property
    def _llm_type(self):
        return "summary"

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.__dict__["calls"] = self.calls + 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.text))])

    async def _astream(self, messages, stop=None, run_manager=None, **kw):
        self.__dict__["calls"] = self.calls + 1
        yield ChatGenerationChunk(message=AIMessageChunk(content=self.text))


def _agentic_history(results: int = 3, chars: int = 30_000) -> list:
    """One request followed by tool calls whose results dwarf the old 4,000-token trim —
    the shape that came back empty and lost a real thread."""
    out = [HumanMessage("earlier request")]
    for i in range(results):
        out.append(AIMessage(content="", tool_calls=[
            {"name": "_dummy_tool", "args": {}, "id": f"h{i}", "type": "tool_call"}]))
        out.append(ToolMessage(content="x" * chars, tool_call_id=f"h{i}", name="_dummy_tool"))
    out.append(AIMessage(content="done with the earlier request"))
    return out


def _summarising_session(summary_model, answer="the answer"):
    """A real graph with the production summariser and steering, over a seeded thread."""
    from langgraph.checkpoint.memory import InMemorySaver
    with mock.patch.object(agent_core, "_SUMMARIZE_AT_TOKENS", 0):
        base = agent_core._middleware()                 # images, steering, clock
    with mock.patch.object(agent_core, "_SUMMARIZE_AT_TOKENS", 1000), \
         mock.patch.object(agent_core, "_SUMMARIZE_KEEP", 2):
        summariser = agent_core._summarization(summary_model)
    graph = agent_core.create_agent(
        model=_ScriptedModel(turns=[{"text": [answer]}]), tools=[_dummy_tool],
        middleware=base + [summariser], checkpointer=InMemorySaver())
    thread = f"t{next(_thread_seq)}"
    config = {"configurable": {"thread_id": thread}}
    graph.update_state(config, {"messages": _agentic_history()})
    return {"graph": graph, "tool_texts": [], "config": config}, thread


@pytest.mark.parametrize("tagged", [True, False])
def test_a_tool_heavy_thread_is_really_summarised_and_none_of_it_reaches_the_card(tagged):
    """Untagged too: the stream filter alone must keep the summary model's tokens and the
    summary message off the card, in case a later change drops the tag."""
    summary = _SummaryModel(tags=["nostream"] if tagged else None)
    session, _ = _summarising_session(summary)
    card = FakeCard()
    with _patch_notify(card, []):
        text = agent_core._stream_to_chat(session, "next question", "oc_1")
    assert text == "the answer"
    assert all("SUMMARY" not in u and "Here is a summary" not in u for u in card.updates)
    head = session["graph"].get_state(session["config"]).values["messages"][0]
    assert agent_core._is_summary(head)
    assert summary.calls >= 1
    assert "SUMMARY-OF-EARLIER-WORK" in head.content
    assert agent_core._SUMMARY_FALLBACK not in head.content


def test_a_steered_message_is_not_echoed_onto_the_card():
    """Steering writes a HumanMessage from a node, which the stream replays."""
    session, thread = _summarising_session(_SummaryModel(tags=["nostream"]))
    agent_core.steer(thread, "STEERED-IN")
    card = FakeCard()
    with _patch_notify(card, []):
        text = agent_core._stream_to_chat(session, "next question", "oc_1")
    assert "STEERED-IN" not in text


def _summary_update(text):
    from langchain_core.messages import RemoveMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), HumanMessage(
        content=f"Here is a summary of the conversation to date:\n\n{text}",
        additional_kwargs={"lc_source": "summarization"})]}


@pytest.mark.parametrize("outcome", ["fallback", "raises"])
def test_a_failed_summary_keeps_the_history_instead_of_replacing_it(outcome):
    """Applying the fallback line is what deleted 114 messages; an exception would fail
    the turn and every one after it. Either way the thread is left as it was."""
    import asyncio
    mw = agent_core._summarization(_SummaryModel())
    base = type(mw).__mro__[1]
    if outcome == "fallback":
        patch = mock.patch.object(base, "abefore_model", mock.AsyncMock(
            return_value=_summary_update(agent_core._SUMMARY_FALLBACK)))
    else:
        patch = mock.patch.object(base, "abefore_model",
                                  mock.AsyncMock(side_effect=RuntimeError("throttled")))
    with patch:
        assert asyncio.run(mw.abefore_model({"messages": []}, None)) is None


def test_a_real_summary_is_applied():
    import asyncio
    mw = agent_core._summarization(_SummaryModel())
    update = _summary_update("what happened so far")
    with mock.patch.object(type(mw).__mro__[1], "abefore_model",
                           mock.AsyncMock(return_value=update)):
        assert asyncio.run(mw.abefore_model({"messages": []}, None)) is update


def test_the_transcript_shows_the_summary_as_a_note_not_as_the_user():
    head = _summary_update("what happened so far")["messages"][1]
    tup = mock.Mock(checkpoint={"channel_values": {"messages": [head, HumanMessage("hi")]}})
    saver = mock.Mock(aget_tuple=mock.AsyncMock(return_value=tup))
    with mock.patch.object(agent_core, "_checkpointer", return_value=saver):
        roles = [m["role"] for m in agent_core.transcript("lark:ou_x")["messages"]]
    assert roles == ["summary", "user"]


# --------------------------- AG-UI (browser path) ----------------------------
# The browser talks to the Runtime directly here, so the one invariant that matters is that
# nothing it sends can name a user: threadId decides whose conversation and whose files.

def test_agui_requests_are_told_apart_from_the_router_protocol():
    import agui
    assert agui.is_agui_request({"threadId": "t", "messages": []}) is True
    assert agui.is_agui_request({"runId": "r"}) is True
    # The router always sends an action, even alongside a thread id.
    assert agui.is_agui_request({"action": "chat", "memorySessionId": "m"}) is False
    assert agui.is_agui_request({}) is False


def test_the_page_is_sent_only_what_it_renders():
    """Snapshots repeat the whole thread and raw events the whole LangGraph payload: a few
    large tool results made one turn's stream 12 MB, which the browser dropped."""
    import agui, json as _j
    from ag_ui.core.events import (EventType, MessagesSnapshotEvent, StateSnapshotEvent,
                                   TextMessageContentEvent, ToolCallResultEvent)
    big = "x" * 100_000
    assert agui._sse(StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot={"m": big})) == b""
    assert agui._sse(MessagesSnapshotEvent(type=EventType.MESSAGES_SNAPSHOT, messages=[])) == b""
    text = agui._sse(TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id="m",
                                             delta="hi", raw_event={"state": big}))
    assert _j.loads(text[len(b"data: "):]) == {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m",
                                               "delta": "hi"}
    result = agui._sse(ToolCallResultEvent(type=EventType.TOOL_CALL_RESULT, message_id="m",
                                           tool_call_id="c", content=big))
    assert b"TOOL_CALL_RESULT" in result and len(result) < 200


def test_the_pages_run_input_is_accepted_with_the_sessions_thread():
    """The shape webui/index.html sends, through the real RunAgentInput: a field it carries
    and we also set must be replaced, not passed twice (TypeError on every web turn)."""
    import agui
    from ag_ui.core.types import RunAgentInput
    page = {"actorId": "web:s-1", "memorySessionId": "sess-x", "runId": "r-1",
            "threadId": "resolved-by-agent", "messages": [{"id": "m-1", "role": "user",
                                                           "content": "hi"}],
            "tools": [], "context": [], "state": {}, "forwardedProps": {}}
    run = RunAgentInput(**agui._run_input_fields(page, "sess-x"))
    assert run.thread_id == "sess-x" and run.run_id == "r-1"


def test_a_web_turn_may_take_as_many_steps_as_a_lark_turn():
    """The AG-UI adapter fills an unset recursion limit with langchain_core's 25, where a Lark
    turn keeps LangGraph's own default; a dozen tool calls must not end a web turn."""
    import agui, asyncio
    from typing import Annotated, TypedDict
    from ag_ui.core.types import RunAgentInput
    from ag_ui_langgraph import LangGraphAgent
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import StateGraph, add_messages

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        n: int

    g = StateGraph(State)
    g.add_node("step", lambda s: {"n": s.get("n", 0) + 1})
    g.add_edge("__start__", "step")
    g.add_conditional_edges("step", lambda s: "step" if s["n"] < 30 else "__end__")
    graph = g.compile(checkpointer=InMemorySaver())
    session = {"config": {"configurable": {"thread_id": "sess-x", "actor_id": "web:s-1"}}}
    run = RunAgentInput(**agui._run_input_fields(
        {"runId": "r-1", "messages": [{"id": "m-1", "role": "user", "content": "hi"}],
         "tools": [], "context": [], "state": {}, "forwardedProps": {}}, "sess-x"))
    adapter = LangGraphAgent(name="t", graph=graph, config=agent_core._run_config(
        session, agent_core.delivery.NoDelivery("test")))

    async def events():
        return [e async for e in adapter.run(run)]
    kinds = [getattr(e, "type", None) for e in asyncio.run(events())]
    assert "RUN_ERROR" not in [str(k).split(".")[-1] for k in kinds], kinds


def test_a_web_turn_stops_at_a_tool_that_needs_a_grant():
    """As a Lark turn does: the model must not go on to narrate a refusal that is certain,
    and the page gets the consent prompt and a finished run."""
    import agui, asyncio
    from types import SimpleNamespace as E
    reached = []

    class Run:
        def __init__(self, **kw): pass

        async def run(self, _input):
            yield E(type="RUN_STARTED")
            yield E(type="TEXT_MESSAGE_CONTENT", delta="我来查看你的文档。")
            yield E(type="TOOL_CALL_START", tool_call_id="t1", tool_call_name="lark_docs")
            reached.append("past the wall")
            yield E(type="TEXT_MESSAGE_CONTENT", delta="还是不行……")

    class Resp:
        def __init__(self, **kw): self.out = b""
        async def prepare(self, _r): pass
        async def write(self, b): self.out += b
        async def write_eof(self): pass

    session = {"mem_sid": "sess-x", "graph": None, "config": {"configurable": {}},
               "tool_idp": {"lark_docs": "lark"}, "consent": {"lark": ""}}
    with mock.patch.object(agent_core, "aget_session", mock.AsyncMock(return_value=session)), \
         mock.patch("ag_ui_langgraph.LangGraphAgent", Run), \
         mock.patch.object(agui, "_sse", lambda e: f"data: {e.type}\n\n".encode()), \
         mock.patch.object(agui.web, "StreamResponse", Resp):
        resp = asyncio.run(agui.handle(None, {"actorId": "web:s-1", "runId": "r-1",
                                              "memorySessionId": "sess-x", "messages": []}, ""))
    frames = [f[len("data: "):] for f in resp.out.decode().split("\n\n") if f]
    assert frames[:2] == ["RUN_STARTED", "TEXT_MESSAGE_CONTENT"]
    assert json.loads(frames[2]) == {"type": "CUSTOM", "name": "consent", "value": {"idp": "lark"}}
    assert json.loads(frames[3])["type"] == "RUN_FINISHED" and len(frames) == 4
    assert reached == [] and agent_core.try_claim("sess-x", "web") is None
    agent_core.release("sess-x")


def test_a_file_the_run_sends_reaches_the_page_on_that_runs_stream():
    import agui, asyncio
    from types import SimpleNamespace as E

    class Run:
        def __init__(self, config=None, **kw):
            self.channel = config["configurable"]["delivery"]

        async def run(self, _input):
            yield E(type="RUN_STARTED")
            yield E(type="TOOL_CALL_START", tool_call_id="t1", tool_call_name="deliver_file")
            self.channel.send_file("chart.png", b"png!", lambda: None)
            yield E(type="TOOL_CALL_RESULT", tool_call_id="t1", content="Sent")
            yield E(type="RUN_FINISHED")

    class Resp:
        def __init__(self, **kw): self.out = b""
        async def prepare(self, _r): pass
        async def write(self, b): self.out += b
        async def write_eof(self): pass

    session = {"mem_sid": "sess-f", "graph": None, "config": {"configurable": {}},
               "tool_idp": {}, "consent": {}}
    with mock.patch.object(agent_core, "aget_session", mock.AsyncMock(return_value=session)), \
         mock.patch("ag_ui_langgraph.LangGraphAgent", Run), \
         mock.patch.object(agui, "_sse", lambda e: f"data: {e.type}\n\n".encode()), \
         mock.patch.object(agui.web, "StreamResponse", Resp):
        resp = asyncio.run(agui.handle(None, {"actorId": "web:s-1", "runId": "r-1",
                                              "memorySessionId": "sess-f", "messages": []}, ""))
    frames = [f[len("data: "):] for f in resp.out.decode().split("\n\n") if f]
    # Written between the events, wherever the producer had got to; before the run ends.
    sent = [json.loads(f) for f in frames if f.startswith("{")]
    assert [f for f in frames if not f.startswith("{")] == [
        "RUN_STARTED", "TOOL_CALL_START", "TOOL_CALL_RESULT", "RUN_FINISHED"]
    assert len(sent) == 1 and frames.index(json.dumps(sent[0])) < frames.index("RUN_FINISHED")
    sent = sent[0]
    assert sent["name"] == "file" and sent["value"]["name"] == "chart.png"
    assert base64.b64decode(sent["value"]["data"]) == b"png!"
    agent_core.release("sess-f")


class _CapturedResp:
    def __init__(self, **kw): self.out = b""
    async def prepare(self, _r): pass
    async def write(self, b): self.out += b
    async def write_eof(self): pass

    def frames(self):
        return [f[len("data: "):] for f in self.out.decode().split("\n\n") if f]


def test_a_message_steers_into_its_own_surfaces_turn_and_only_that():
    agent_core._pending_steer.clear()
    assert agent_core.claim_or_steer("t-cs", "web", "web:tab-a", "first") is None
    try:
        assert agent_core.claim_or_steer("t-cs", "web", "web:tab-a", "also this", "m2") == "steered"
        # Another tab, a reload or another device would have the answer stream into this one.
        assert agent_core.claim_or_steer("t-cs", "web", "web:tab-b", "from b") == "web-other"
        assert agent_core.claim_or_steer("t-cs", "web", "", "no stream id") == "web-other"
        assert agent_core.claim_or_steer("t-cs", "lark", "oc_1", "from lark") == "web"
        assert [q[:2] for q in agent_core._pending_steer["t-cs"]] == [("also this", "m2")]
        # Taken as a continuation, the thread stays claimed; with nothing left it is freed.
        assert [q[0] for q in agent_core.take_steer_or_release("t-cs")] == ["also this"]
        assert agent_core.thread_owner("t-cs") == "web"
        assert agent_core.take_steer_or_release("t-cs") == []
        assert agent_core.thread_owner("t-cs") is None
    finally:
        agent_core.release("t-cs")


def test_a_message_sent_from_the_page_mid_turn_joins_it_and_is_answered_below():
    """As in Lark chat: what the page sends while its turn runs is not refused as busy. Too
    late for the running model call, it is answered as a continuation on the same stream,
    marked so the page starts the reply under that message."""
    import agui, asyncio
    from types import SimpleNamespace as E
    agent_core._pending_steer.clear()
    inputs = []

    class Run:
        def __init__(self, **kw): pass

        async def run(self, run_input):
            inputs.append([m.content for m in run_input.messages])
            yield E(type="RUN_STARTED")
            if len(inputs) == 1:      # the page sends a second message meanwhile
                resp = _CapturedResp()
                with mock.patch.object(agui.web, "StreamResponse", lambda **kw: resp):
                    await agui.handle(None, {**page, "messages": [
                        {"id": "m2", "role": "user", "content": "also this"}]}, "")
                frames = resp.frames()
                assert [json.loads(f)["type"] for f in frames] == ["RUN_STARTED", "CUSTOM", "RUN_FINISHED"]
                assert json.loads(frames[1])["name"] == "queued"
                # Another tab is told, not steered: its answer would appear in this one.
                other = _CapturedResp()
                with mock.patch.object(agui.web, "StreamResponse", lambda **kw: other):
                    await agui.handle(None, {**page, "forwardedProps": {"streamId": "tab-b"},
                                             "messages": [{"id": "m3", "role": "user",
                                                           "content": "from b"}]}, "")
                assert "另一个窗口" in other.out.decode()
            yield E(type="RUN_FINISHED")

    session = {"mem_sid": "sess-st", "graph": None, "config": {"configurable": {}},
               "tool_idp": {}, "consent": {}}
    page = {"actorId": "web:s-1", "memorySessionId": "sess-st",
            "forwardedProps": {"streamId": "tab-a"}}
    resp = _CapturedResp()
    with mock.patch.object(agent_core, "aget_session", mock.AsyncMock(return_value=session)), \
         mock.patch("ag_ui_langgraph.LangGraphAgent", Run), \
         mock.patch.object(agui, "_sse", lambda e: f"data: {e.type}\n\n".encode()), \
         mock.patch.object(agui.web, "StreamResponse", lambda **kw: resp):
        asyncio.run(agui.handle(None, {**page, "runId": "r-1", "messages": [
            {"id": "m1", "role": "user", "content": "first"}]}, ""))
    frames = resp.frames()
    assert inputs == [["first"], ["also this"]]
    assert frames[:2] == ["RUN_STARTED", "RUN_FINISHED"] and frames[-1] == "RUN_FINISHED"
    assert json.loads(frames[2])["name"] == "steered" and frames[3] == "RUN_STARTED"
    assert agent_core.thread_owner("sess-st") is None and "sess-st" not in agent_core._turn_cards


def test_a_message_steered_into_a_turn_that_stops_for_consent_is_reported_not_lost():
    import agui, asyncio
    from types import SimpleNamespace as E
    agent_core._pending_steer.clear()

    class Run:
        def __init__(self, **kw): pass

        async def run(self, _input):
            yield E(type="RUN_STARTED")
            assert agent_core.claim_or_steer("sess-w", "web", "web:tab-a", "also", "m2") == "steered"
            yield E(type="TOOL_CALL_START", tool_call_id="t1", tool_call_name="lark_docs")

    session = {"mem_sid": "sess-w", "graph": None, "config": {"configurable": {}},
               "tool_idp": {"lark_docs": "lark"}, "consent": {"lark": ""}}
    resp = _CapturedResp()
    with mock.patch.object(agent_core, "aget_session", mock.AsyncMock(return_value=session)), \
         mock.patch("ag_ui_langgraph.LangGraphAgent", Run), \
         mock.patch.object(agui, "_sse", lambda e: f"data: {e.type}\n\n".encode()), \
         mock.patch.object(agui.web, "StreamResponse", lambda **kw: resp):
        asyncio.run(agui.handle(None, {"actorId": "web:s-1", "memorySessionId": "sess-w",
                                       "forwardedProps": {"streamId": "tab-a"}, "runId": "r",
                                       "messages": []}, ""))
    marks = [json.loads(f) for f in resp.frames() if f.startswith("{")]
    assert {"type": "CUSTOM", "name": "unanswered", "value": {"count": 1}} in marks
    assert agent_core.thread_owner("sess-w") is None


def test_a_web_run_never_releases_a_claim_it_no_longer_holds():
    """The run releases in the same step as finding nothing left; releasing again when the
    producer ends could free a turn that claimed the thread in between."""
    import agui
    state = {"released": True}
    agent_core._track(+1, "sess-r", "lark", "oc_1")      # someone else's turn, meanwhile
    try:
        assert agui._end("sess-r", agui._PageMarks([]), state) == []
        assert agent_core.thread_owner("sess-r") == "lark"
    finally:
        agent_core.release("sess-r")


def test_a_delivered_file_is_stored_under_its_owner_and_only_a_picture_is_shown_inline():
    import deliveries, hashlib
    s3 = mock.Mock()
    s3.generate_presigned_url.side_effect = lambda op, ExpiresIn, Params: f"https://s/{Params['Key']}"
    with mock.patch.object(deliveries, "_BUCKET", "b"), mock.patch.object(deliveries, "_s3", return_value=s3):
        key = deliveries.put("web:s-1", "../chart.png", b"x")
        owner = hashlib.sha256(b"web:s-1").hexdigest()[:32]
        assert key.startswith(f"deliveries/{owner}/") and key.endswith("/chart.png")
        put = s3.put_object.call_args.kwargs
        assert put["ContentType"] == "image/png" and put["ContentDisposition"].startswith("inline")
        deliveries.put("web:s-1", "page.svg", b"<svg/>")
        svg = s3.put_object.call_args.kwargs
        assert svg["ContentType"] == "application/octet-stream"
        assert svg["ContentDisposition"].startswith("attachment")
        assert deliveries.url(key, "chart.png", "web:s-1") == f"https://s/{key}"
        # Only this actor's own deliveries are signed or deleted.
        assert deliveries.url(key, "chart.png", "web:someone-else") == ""
        assert deliveries.url("state/x", "x.png", "web:s-1") == ""
        assert deliveries.delete(["state/x", key, key], "web:s-1") == 1
        assert deliveries.delete([key], "web:someone-else") == 0


def test_the_history_shows_a_delivered_file_again_and_clear_removes_it():
    from langchain_core.messages import AIMessage as _AI, ToolMessage as _Tool
    from langgraph.checkpoint.memory import InMemorySaver
    saver = InMemorySaver()
    graph = agent_core.create_agent(model=_ScriptedModel(turns=[{"text": ["ok"]}]),
                                    tools=[_dummy_tool], checkpointer=saver)
    cfg = {"configurable": {"thread_id": "t-files"}}
    sent = {"file": {"key": "deliveries/o/u/chart.png", "name": "chart.png", "size": 3}}
    agent_core._run(graph.aupdate_state(cfg, {"messages": [
        HumanMessage("chart please", id="h1"),
        _AI(content="", id="a1", tool_calls=[{"name": "deliver_file", "args": {}, "id": "c1",
                                              "type": "tool_call"}]),
        _Tool(content="Sent chart.png to the page", tool_call_id="c1", artifact=sent),
        _AI(content="here it is", id="a2")]}, as_node="model"))
    with mock.patch.object(agent_core, "_checkpointer", return_value=saver), \
         mock.patch.object(agent_core, "_thread_config", return_value=cfg), \
         mock.patch.object(agent_core.deliveries, "url", return_value="https://fresh") as url, \
         mock.patch.object(agent_core.deliveries, "delete") as delete:
        files = [m for m in agent_core.transcript("web:s-1")["messages"] if m["role"] == "file"]
        assert files == [{"role": "file", "name": "chart.png", "size": 3, "url": "https://fresh"}]
        url.assert_called_once_with("deliveries/o/u/chart.png", "chart.png", "web:s-1")
        assert agent_core.clear_history("web:s-1")["deleted"] is True
        assert delete.call_args.args[0][0] == "deliveries/o/u/chart.png"
        assert delete.call_args.args[1] == "web:s-1"


def test_a_picture_from_the_page_is_seen_on_its_turn_and_kept_as_a_placeholder():
    """As a Lark picture is: inlined into the checkpoint it would be re-billed on every turn."""
    import agui, asyncio, io
    from types import SimpleNamespace as E
    from PIL import Image
    buf = io.BytesIO(); Image.new("RGB", (4, 4), "red").save(buf, format="PNG")
    png = base64.b64encode(buf.getvalue()).decode()
    seen = {}

    class Run:
        def __init__(self, **kw): pass

        async def run(self, run_input):
            m = run_input.messages[-1]
            seen["message"] = (m.id, m.content)
            seen["pixels"] = agent_core.images.pixels_for("sess-img")
            seen["handles"] = agent_core.images.handles_for("sess-img")
            yield E(type="RUN_STARTED")
            yield E(type="RUN_FINISHED")

    session = {"mem_sid": "sess-img", "graph": None, "config": {"configurable": {}},
               "tool_idp": {}, "consent": {}}
    with mock.patch.object(agent_core, "aget_session", mock.AsyncMock(return_value=session)), \
         mock.patch("ag_ui_langgraph.LangGraphAgent", Run), \
         mock.patch.object(agui, "_sse", lambda e: f"data: {e.type}\n\n".encode()), \
         mock.patch.object(agui.web, "StreamResponse", lambda **kw: _CapturedResp()):
        asyncio.run(agui.handle(None, {"actorId": "web:s-1", "memorySessionId": "sess-img",
                                       "runId": "r", "messages": [{"id": "page-1", "role": "user",
            "content": [{"type": "text", "text": "what is this"},
                        {"type": "image", "source": {"type": "data", "value": png,
                                                     "mimeType": "image/png"}}]}]}, ""))
    mid, content = seen["message"]
    assert content == "what is this\n[image 1: image-1.png]" and mid.startswith("msg-")
    assert [b["type"] for b in seen["pixels"][mid]] == ["image"]
    assert seen["handles"][0]["data"] == buf.getvalue()          # save_attachment's source
    assert agent_core.images.pixels_for("sess-img") == {}         # gone with the turn


def test_a_picture_the_model_cannot_read_is_kept_but_not_shown_to_it():
    """Labelled as a jpeg, an unreadable picture would make Bedrock refuse the whole turn."""
    text, handles, blocks = agent_core.images.from_page([
        {"type": "text", "text": "look"},
        {"type": "image", "source": {"type": "data", "value": base64.b64encode(b"not an image").decode(),
                                     "mimeType": "image/heic"}}])
    assert text == "look" and blocks == [] and handles[0]["name"] == "image-1.heic"
    marks, _ = agent_core.images.register("t-heic", handles, blocks)
    assert "not a picture the model can read" in marks and "save_attachment" in marks
    agent_core.images.end_turn("t-heic")


def test_a_web_run_stamps_an_unanswered_message_only_when_it_ended_without_a_reply():
    """Stopped at a wall, or failed: stamped with that run's own message's arrival, while the
    claim is held. A run that replied is left to the clock and costs no extra read."""
    import agui, asyncio
    from types import SimpleNamespace as E
    agent_core._pending_steer.clear()
    stamped, owners = [], []

    async def fake_stamp(graph, config, at_ms):
        stamped.append(at_ms)
        owners.append(agent_core.thread_owner("sess-sm"))
        return 1

    def run_with(script):
        class Run:
            def __init__(self, **kw): pass

            async def run(self, _input):
                for ev in script.pop(0):
                    if callable(ev):
                        ev()
                    else:
                        yield ev
        session = {"mem_sid": "sess-sm", "graph": object(), "config": {"configurable": {}},
                   "tool_idp": {"lark_docs": "lark"}, "consent": {"lark": ""}}
        stamped.clear(); owners.clear()
        with mock.patch.object(agent_core, "aget_session", mock.AsyncMock(return_value=session)), \
             mock.patch.object(agent_core, "astamp_unanswered", fake_stamp), \
             mock.patch("ag_ui_langgraph.LangGraphAgent", Run), \
             mock.patch.object(agui, "_sse", lambda e: f"data: {e.type}\n\n".encode()), \
             mock.patch.object(agui.web, "StreamResponse", lambda **kw: _CapturedResp()):
            asyncio.run(agui.handle(None, {"actorId": "web:s-1", "memorySessionId": "sess-sm",
                                           "forwardedProps": {"streamId": "tab"}, "runId": "r",
                                           "messages": []}, ""))
        return list(stamped), list(owners)

    ok = [E(type="RUN_STARTED"), E(type="RUN_FINISHED")]
    assert run_with([ok]) == ([], [])
    at, held = run_with([[E(type="RUN_STARTED"), E(type="RUN_ERROR")]])
    assert len(at) == 1 and held == ["web"]
    at, held = run_with([[E(type="RUN_STARTED"),
                          E(type="TOOL_CALL_START", tool_call_id="t", tool_call_name="lark_docs")]])
    assert len(at) == 1 and held == ["web"]
    # A continuation stopped at the wall: stamped with the steered message's own arrival.
    sent = {}

    def steer():
        time.sleep(0.01)       # so the steered message arrives measurably after the first
        sent["ms"] = int(time.time() * 1000)
        agent_core.steer("sess-sm", "also", "m2")
    at, held = run_with([[E(type="RUN_STARTED"), steer, E(type="RUN_FINISHED")],
                         [E(type="RUN_STARTED"),
                          E(type="TOOL_CALL_START", tool_call_id="t", tool_call_name="lark_docs")]])
    assert held == ["web"] and len(at) == 1 and at[0] >= sent["ms"]
    assert agent_core.thread_owner("sess-sm") is None


def test_a_refused_tool_on_the_page_ends_the_run_with_a_consent_prompt():
    """The page gets the system to authorise, not a link: it mints one through /auth when the
    user clicks, with the token it will complete with."""
    import agui
    session = {"tool_texts": [("google_whoami", "no user token (authorize first)")],
               "tool_idp": {"google_whoami": "google"}, "consent": {"google": ""}}
    event = json.loads(agui._consent(session).decode()[len("data: "):])
    assert event == {"type": "CUSTOM", "name": "consent", "value": {"idp": "google"}}
    assert agui._consent({"tool_texts": [], "tool_idp": {}, "consent": {}}) == b""


def test_agui_takes_its_thread_from_the_session_not_the_pages_thread_id():
    """AG-UI's own threadId is the page's to choose, so it is replaced by the session's —
    which is built on the page's memorySessionId only after server.py checked it belongs to
    the verified actor."""
    import agui, inspect
    src = inspect.getsource(agui.handle)
    assert 'thread_id = payload.get("memorySessionId") or agent_core.thread_for(actor_id)' in src
    assert "_pictures_handed_over, payload, thread_id)" in src and "_run_input_fields(handed, thread_id)" in src
    assert agui._run_input_fields({"threadId": "theirs"}, "ours")["threadId"] == "ours"


# --------------------------- who is calling ----------------------------------
# Any allowlisted user can take their own token from the web page and call the Runtime
# directly, so the actor comes from a verified bearer and the payload may only agree with it.

_POOL_ISSUER = "https://cognito-idp.ap-northeast-1.amazonaws.com/pool-1"


@pytest.fixture
def _signer():
    """A key pair standing in for the pool's, and a way to mint tokens with it."""
    import jwt as pyjwt, time as _t, caller
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    keys = mock.Mock()
    keys.get_signing_key_from_jwt.return_value = mock.Mock(key=key.public_key())

    def mint(signing_key=key, **over):
        claims = {"iss": _POOL_ISSUER, "token_use": "access", "client_id": "client-1",
                  "username": "lark:ou_alice", "sub": "sub-alice",
                  "exp": int(_t.time()) + 600, **over}
        return "Bearer " + pyjwt.encode(claims, signing_key, algorithm="RS256")
    with mock.patch.object(caller, "_ISSUER", _POOL_ISSUER), \
         mock.patch.object(caller, "_CLIENT", "client-1"), \
         mock.patch.object(caller, "_WEB_CLIENT", "web-client"), \
         mock.patch.object(caller, "_keys", return_value=keys):
        yield mint


def test_a_web_users_account_comes_from_the_verified_token_and_nothing_else(_signer):
    import caller, server
    web = _signer(client_id="web-client", username="carol@example.com", sub="s-carol")
    assert caller.identify([web]) == ("web:s-carol", "carol@example.com")
    assert caller.identify([_signer()]) == ("lark:ou_alice", "")     # Lark's comes from Lark
    payload = {"actorAccount": "mallory@example.com"}
    req = mock.Mock()
    req.headers.getall.return_value = [web]
    assert server._bind_caller(req, payload) is None
    assert payload["actorAccount"] == "carol@example.com"


def test_whoami_says_how_they_signed_in_as_whom_and_what_they_granted():
    with mock.patch.object(agent_core, "auth_status", return_value={
            "lark": True, "google": False, "accounts": {"lark": "XUESONG"}}):
        tool = agent_core._whoami_tool("web:s-carol", "wat")
        out = json.loads(tool.invoke({}, config={"configurable": {"account": "carol@example.com"}}))
        assert out == {"signed_in_with": "the web page (its own sign-in)",
                       "account": "carol@example.com", "user_id": "web:s-carol",
                       "authorised": {"lark": "XUESONG"}, "not_authorised": ["google"]}
        lark = json.loads(agent_core._whoami_tool("lark:ou_alice", "wat").invoke({}))
        assert lark["signed_in_with"] == "Lark" and lark["account"] == "XUESONG"
    # The account is never a parameter the model could fill in.
    assert "account" not in (tool.args or {})


def test_the_actor_is_the_username_of_a_verified_access_token(_signer):
    import caller, time as _t
    from cryptography.hazmat.primitives.asymmetric import rsa
    assert caller.actor([_signer()]) == "lark:ou_alice"
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    for bad in (_signer(signing_key=other_key),          # not the pool's signature
                _signer(client_id="someone-else"),       # another app's token
                _signer(token_use="id"),                 # an ID token
                _signer(iss="https://evil.example"),
                _signer(exp=int(_t.time()) - 60)):
        assert caller.actor([bad]) == "", bad
    # Which of two headers the platform checked is unknown, so two are refused outright.
    assert caller.actor([_signer(), _signer(username="lark:ou_bob")]) == ""
    assert caller.actor([]) == "" and caller.actor(["Basic abc"]) == ""
    # A container clock a little behind Cognito's must not lock out a token minted just now.
    assert caller.actor([_signer(iat=int(_t.time()) + 30)]) == "lark:ou_alice"
    assert caller.actor(["bearer" + _signer()[len("Bearer"):]]) == "lark:ou_alice"


def test_the_issuing_client_decides_the_actors_namespace(_signer):
    """Cognito signs client_id, so a caller cannot choose it. A web user is `web:{sub}`, and
    the router's client yields only `lark:` actors, so a stray user of it is nobody."""
    import caller
    web = dict(client_id="web-client", sub="sub-carol")
    assert caller.actor([_signer(**web, username="carol@example.com")]) == "web:sub-carol"
    # A Lark user's own account signed in at the hosted login carries their sub: refused.
    assert caller.actor([_signer(**web, username="lark:ou_victim")]) == ""
    assert caller.actor([_signer(username="service:websearch")]) == ""
    with mock.patch.object(caller, "_WEB_CLIENT", ""):          # no web login deployed
        assert caller.actor([_signer(**web)]) == ""


def _bind(headers, payload):
    import server
    from multidict import CIMultiDict
    request = mock.Mock(headers=CIMultiDict(headers))
    return server._bind_caller(request, payload)


def test_a_payload_may_only_agree_with_the_verified_caller(_signer):
    import agent_core
    alice = agent_core._session_id_for("lark:ou_alice")
    payload = {"action": "chat", "actorId": "lark:ou_alice", "userId": "user_abc",
               "memorySessionId": alice}
    assert _bind([("Authorization", _signer())], payload) is None
    # The router's userId is an internal key, never an actor, so it is not compared.
    assert payload["actorId"] == "lark:ou_alice"
    # Absent, the actor is filled in from the token — the A2A adapter names nobody.
    payload = {"action": "chat"}
    assert _bind([("Authorization", _signer())], payload) is None
    assert payload["actorId"] == "lark:ou_alice"
    for payload, status in (({"action": "chat", "actorId": "lark:ou_bob"}, 403),
                            ({"action": "history_stats"}, None)):
        r = _bind([("Authorization", _signer())], payload)
        assert (r.status if r else None) == status
    assert _bind([], {"action": "chat"}).status == 401


def test_a_thread_is_honoured_only_under_the_callers_own_prefix(_signer):
    """The checkpoint is keyed by thread id alone and the default id is computable by
    anyone, so naming another user's thread under your own identity has to fail too."""
    import agent_core
    bob = agent_core._session_id_for("lark:ou_bob")
    alice = agent_core._session_id_for("lark:ou_alice")
    for thread in (bob, bob + "-0123456789ab", "sess-" + "0" * 32):
        r = _bind([("Authorization", _signer())], {"action": "clear_history",
                                                   "memorySessionId": thread})
        assert r is not None and r.status == 403, thread
    assert agent_core.thread_belongs("lark:ou_alice", alice)
    assert agent_core.thread_belongs("lark:ou_alice", alice + "-0123456789ab")
    assert not agent_core.thread_belongs("lark:ou_alice", alice + "-xyz")
    assert not agent_core.thread_belongs("lark:ou_alice", alice + "-0123456789abc")


def test_a_turn_is_answered_only_into_a_chat_the_caller_is_in():
    """The bot's token reaches every chat the bot is in, so a verified caller naming some
    other chat — or another chat's message to pull a picture from — is refused."""
    import server
    member = lambda chat, who: (chat, who) == ("oc_mine", "ou_alice")
    chat_of = {"om_mine": "oc_mine", "om_theirs": "oc_theirs"}.get
    with mock.patch.object(server.lark_notify, "in_chat", side_effect=member), \
         mock.patch.object(server.lark_notify, "chat_of", side_effect=lambda m: chat_of(m, "")):
        ok = lambda chat, imgs=(): server._chat_refusal("lark:ou_alice", chat, list(imgs))
        assert ok("oc_mine") == "" and ok("ou_alice") == ""
        assert ok("oc_theirs") == "foreign_chat" and ok("ou_bob") == "foreign_chat"
        assert ok("oc_mine", [{"message_id": "om_mine", "key": "k"}]) == ""
        assert ok("oc_mine", [{"message_id": "om_theirs", "key": "k"}]) == "foreign_message"
        assert ok("oc_mine", [{"key": "k"}]) == "foreign_message"
        # The message being answered is reacted to with the bot's token: it must be there too.
        msg = lambda mid: server._chat_refusal("lark:ou_alice", "oc_mine", [], mid)
        assert msg("om_mine") == "" and msg("om_theirs") == "foreign_message"


def test_an_id_from_a_request_cannot_address_another_endpoint():
    """`messages/{id}?` is the recall endpoint, so a reaction id carrying `?` would have
    recalled any message the bot sent; a key carrying `../` would read another chat's."""
    sys.path.insert(0, os.path.dirname(__file__))
    import importlib, lark_notify
    importlib.reload(lark_notify)
    calls = []
    with mock.patch.object(lark_notify, "_tenant_token", return_value="t"), \
         mock.patch.object(lark_notify, "_call", side_effect=lambda *a, **k: calls.append(a) or {"code": 0}):
        assert lark_notify.remove_reaction("om_victim?", "x") is False
        assert lark_notify.remove_reaction("om_1", "../chats/oc_x") is False
        assert lark_notify.remove_reaction("om_1", "ZCaCIjUBVVW-rm5L_3Z=") is True
    assert len(calls) == 1
    with mock.patch.object(lark_notify, "_tenant_token", return_value="t"):
        with pytest.raises(lark_notify.LarkError):
            lark_notify.fetch_message_resource("om_mine", "../../om_theirs/resources/img_x")
    assert lark_notify.chat_of("om_1/../om_2") == ""


def test_membership_is_read_from_lark_and_cached_only_when_true():
    sys.path.insert(0, os.path.dirname(__file__))
    import importlib, lark_notify
    importlib.reload(lark_notify)
    pages = [{"code": 0, "data": {"items": [{"member_id": "ou_x"}], "has_more": True,
                                  "page_token": "p2"}},
             {"code": 0, "data": {"items": [{"member_id": "ou_alice"}], "has_more": False}}]
    with mock.patch.object(lark_notify, "_get", side_effect=pages) as get:
        assert lark_notify.in_chat("oc_1", "ou_alice") is True
        assert lark_notify.in_chat("oc_1", "ou_alice") is True      # cached
    assert get.call_count == 2
    with mock.patch.object(lark_notify, "_get", return_value={"code": 1, "msg": "no"}):
        assert lark_notify.in_chat("oc_2", "ou_alice") is False


def test_only_the_probes_go_without_a_bearer():
    import server, inspect
    src = inspect.getsource(server.handle_invocations)
    gate = src.index("_bind_caller, request, payload")
    assert src.index('if action == "status"') > gate and src.index("if is_agui:") > gate
    assert 'action not in ("status", "warmup")' in src


def test_a_session_the_page_talks_on_counts_as_served_from_its_first_turn():
    """/status reported 「服务本会话 0 秒」 for a session that had just served a web turn: the
    AG-UI branch returned before the session was recorded."""
    import server, inspect
    src = inspect.getsource(server.handle_invocations)
    assert src.index("_session_first_seen.setdefault(sid") < src.index("if is_agui:")


def test_consent_is_asked_for_the_system_the_tool_acts_on():
    """A Google grant without a Lark one was walled with the Lark card for google_whoami:
    the wall keyed on Lark alone. Each tool now walls only on its own system's grant."""
    tool_idp = {"lark_docs": "lark", "google_whoami": "google"}
    s = {"consent": {"lark": "https://lark"}, "tool_idp": tool_idp}
    assert agent_core._walled_idp(s, "lark_docs") == "lark"
    assert agent_core._walled_idp(s, "google_whoami") is None      # Google is granted
    assert agent_core._walled_idp(s, "memory_save") is None        # untagged never walls
    s = {"consent": {"google": ""}, "tool_idp": tool_idp}
    assert agent_core._walled_idp(s, "google_whoami") == "google"
    assert agent_core._walled_idp(s, "lark_docs") is None


def test_stream_walls_a_google_tool_with_the_google_card():
    card = FakeCard()
    session = _session_for(["查一下…"], tool_calls=["google_whoami"], follow_up="没有权限",
                           consent={"google": ""}, tool_idp={"google_whoami": "google"})
    with _patch_notify(card, []):
        text = agent_core._stream_to_chat(session, "我的 Google 账号", "oc_1")
    assert session.get("walled_tool") == "google_whoami"
    assert "Google" in text and "Lark" not in text


def test_downstream_consent_is_recorded_at_build_and_minted_only_when_reached():
    """Minting at build would start a consent flow nobody asked for and kill any link the
    user is already holding; the build only peeks, the wall mints."""
    async def fake_open(stack, connection):
        return [_make_tool("google_whoami")] if connection["url"] else [_dummy_tool]

    with mock.patch.dict(agent_core._DOWNSTREAMS, {"google": ("https://g", "p", ["openid"])},
                         clear=True), \
         mock.patch.object(agent_core.lark_3lo, "get_user_lark_token",
                           return_value=("token", "lark-tok")), \
         mock.patch.object(agent_core.lark_3lo, "peek_user_token_for", return_value=""), \
         mock.patch.object(agent_core.lark_3lo, "get_user_token_for",
                           return_value=("auth_url", "https://google-consent")) as mint, \
         mock.patch.object(agent_core.lark_3lo, "mcp_connection_for",
                           side_effect=lambda tok, url="": {"url": url}), \
         mock.patch.object(agent_core, "_aopen_tools", fake_open), \
         mock.patch.object(agent_core.websearch, "available", return_value=False):
        s = agent_core._build_session("lark:ou_x", "", "mem1")
        assert s["consent"] == {"google": ""} and not s["auth_url"]
        assert s["tool_idp"]["google_whoami"] == "google"
        mint.assert_not_called()
        assert agent_core._consent_link(s, "google") == "https://google-consent"
        mint.assert_called_once()
