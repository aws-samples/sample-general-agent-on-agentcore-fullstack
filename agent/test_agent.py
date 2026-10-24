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
import os
import re
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(__file__))
# What a deploy injects; the modules read it at import and have no fallback.
os.environ.setdefault("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-5")
os.environ.setdefault("LARK_API_DOMAIN", "https://open.larksuite.com")

import agent_core

# Message classes are used throughout; agent_core already pulls them in.
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ------------------------------- identity -----------------------------------

def test_derive_password_deterministic_and_complex():
    import identity
    with mock.patch.object(identity, "_get_salt", return_value="test-salt"):
        p1 = identity._derive_password("lark:ou_abc")
        p2 = identity._derive_password("lark:ou_abc")
        p3 = identity._derive_password("lark:ou_xyz")
    assert p1 == p2                       # deterministic
    assert p1 != p3                       # per-user
    assert p1.endswith("Aa1!")            # complexity suffix
    assert len(p1) == 36


def test_jwt_exp_parses_unverified():
    import identity
    payload = base64.urlsafe_b64encode(json.dumps({"exp": 1234567890}).encode()).decode().rstrip("=")
    token = f"header.{payload}.sig"
    assert identity._jwt_exp(token) == 1234567890.0


def test_jwt_exp_bad_token_returns_zero():
    import identity
    assert identity._jwt_exp("not-a-jwt") == 0.0


def test_ensure_user_email_sanitizes_colon():
    """open_id-based username 'lark:ou_x' must not produce an invalid email."""
    import identity
    captured = {}

    def fake_create(**kw):
        captured["email"] = next(a["Value"] for a in kw["UserAttributes"] if a["Name"] == "email")

    with mock.patch.object(identity, "_cognito") as c:
        from botocore.exceptions import ClientError
        c.admin_get_user.side_effect = ClientError(
            {"Error": {"Code": "UserNotFoundException"}}, "AdminGetUser")
        c.admin_create_user.side_effect = fake_create
        c.admin_set_user_password.return_value = {}
        with mock.patch.object(identity, "_get_salt", return_value="s"):
            identity._ensure_user("lark:ou_abc", "")
    assert ":" not in captured["email"]           # colon replaced
    assert captured["email"] == "lark-ou_abc@lark.local"


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
    agent_core.remember_thread("lark:ou_x", "sess-rotated-by-reset")
    assert agent_core.thread_for("lark:ou_x") == "sess-rotated-by-reset"
    # An empty value must not erase what the router said.
    agent_core.remember_thread("lark:ou_x", "")
    assert agent_core.thread_for("lark:ou_x") == "sess-rotated-by-reset"
    # And it is per actor.
    assert agent_core.thread_for("lark:ou_other") == agent_core._session_id_for("lark:ou_other")


def test_the_page_sends_no_thread_id():
    """It cannot be trusted with one, so it must not be in a position to send one."""
    import pathlib as _p
    page = (_p.Path(__file__).parents[1] / "webui" / "index.html").read_text()
    assert "memorySessionId" not in page
    assert "runtimeSessionId" in page      # the routing key, which is not a capability


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
    assert agent_core._pending_steer == {"t-a": [("for A", "", False)]}


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
    """websearch imports mcp (not installable here), so assert the two things that
    silently break the Gateway call: the protocol-version header, without which it
    negotiates 2025-03-26 and rejects with -32022, and an access token rather than
    an ID token, since the Gateway checks the client_id claim."""
    src = open(os.path.join(os.path.dirname(__file__), "websearch.py"), encoding="utf-8").read()
    assert "MCP-Protocol-Version" in src
    assert "2025-11-25" in src
    assert "get_user_jwt" in src        # identity.py returns the ACCESS token
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
        mock.Mock(StreamingCard=lambda chat_id: card,
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

    def close(self, final_text):
        self.closed_with = final_text
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
        if content is None and "elements" in body:       # the answer's first write
            content = json.loads(body["elements"])[0]["content"]
        # Only the answer's text is under test here; status writes are counted elsewhere.
        if "/elements/status" in url or url.endswith("/settings"):
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
    card.card_id, card.ok = "c1", True
    return lark_notify, card, writes


def test_the_answer_appears_above_the_status_line_then_streams_in_place():
    """CardKit rejects an empty text update, so the answer element does not exist until
    there are words for it — and it goes above the status, which stays at the bottom."""
    ln, card, writes = _recording_card()
    card._write("我先")
    card._write("我先看一下")
    (m1, p1, b1), (m2, p2, b2) = writes
    assert (m1, p1, b1["type"], b1["target_element_id"]) == (
        "POST", "/elements", "insert_before", "status")
    assert (m2, p2, b2["content"]) == ("PUT", "/elements/answer/content", "我先看一下")


def test_the_tool_count_comes_from_results_not_from_starts():
    """A streamed tool call names its tool on more than one chunk, so counting starts
    would overcount; a result is one call that really ran."""
    ln, card, _ = _recording_card()
    card.tool_started(); card.tool_started(); card.tool_finished()
    title, steps = card.summary()
    assert "调用 1 次工具" in title
    assert any(s.startswith("调用工具") for s in steps)


def test_a_turn_without_tools_says_only_how_long_it_took():
    ln, card, _ = _recording_card()
    title, _ = card.summary()
    assert title.startswith("⚙️ 执行过程 · 耗时") and "工具" not in title


def test_closing_turns_the_status_into_a_collapsed_panel():
    ln, card, writes = _recording_card()
    card.update("答案")
    assert card.close("答案") is True
    panel = [b for m, p, b in writes if m == "PUT" and p == "/elements/status"]
    assert panel, "the status line must be replaced by the summary panel"
    el = json.loads(panel[-1]["element"])
    assert el["tag"] == "collapsible_panel" and el["expanded"] is False
    assert writes[-1][1] == "/settings"          # streaming turned off last


def test_a_failed_panel_write_does_not_cost_the_answer():
    """The panel is a footnote. If CardKit refuses it the answer is still delivered and
    the card is still closed, so the caller must not fall back to posting text again."""
    ln, card, writes = _recording_card()
    orig = ln._call
    ln._call = lambda m, u, b, bearer="": ({"code": 1, "msg": "no"}
                                           if u.endswith("/elements/status") and m == "PUT"
                                           else orig(m, u, b, bearer))
    assert card.close("答案") is True


def test_the_heartbeat_moves_the_seconds_only_while_waiting():
    """A long tool call emits no event, so the status is refreshed on a timer while the
    turn waits — and not once the answer is streaming, where it would only add writes."""
    import threading, time as _t
    ln, card, writes = _recording_card()
    ln._HEARTBEAT_SECONDS = 0.05
    card._worker = threading.Thread(target=card._pump, daemon=True)
    card._worker.start()
    card.tool_started()
    _t.sleep(1.3)  # nosemgrep: arbitrary-sleep — lets a few heartbeats land
    waiting = sum(1 for m, p, _ in writes if p == "/elements/status/content")
    assert waiting >= 2, "the seconds must move during a tool call"
    card.update("开始回答")
    _t.sleep(1.3)  # nosemgrep: arbitrary-sleep — would expose a heartbeat that kept going
    card._stop.set(); card._wake.set(); card._worker.join(timeout=2)
    after = [b["content"] for m, p, b in writes if p == "/elements/status/content"][waiting:]
    assert all("撰写" in c for c in after), f"heartbeat kept running while writing: {after}"


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


def test_the_transcript_action_does_not_believe_the_claimed_actor():
    """A thread is addressed by the actor alone, so trusting the claim would hand any valid
    token holder somebody else's conversation. The session build is the check."""
    import inspect
    import server
    src = inspect.getsource(server.handle_invocations)
    i = src.index('action == "transcript"')
    block = src[i:i + 900]
    assert "aget_session" in block and 'session["mem_sid"]' in block
    assert "identity_error" in block and "auth_url" in block


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


def test_summarisation_can_be_disabled_outright():
    """Steering and image attachment stay either way — one keeps a mid-turn message from
    being lost, the other is what lets the model see a screenshot at all."""
    with mock.patch.object(agent_core, "_SUMMARIZE_AT_TOKENS", 0):
        assert [type(m).__name__ for m in agent_core._middleware()] == [
            "ImageMiddleware", "SteeringMiddleware"]


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
    agent_core._track(+1, "t-busy", "lark")
    try:
        with mock.patch.object(agent_core.images, "blocks_for", return_value=[]), \
             mock.patch.object(agent_core, "lark_notify"):
            out = agent_core.chat_async("lark:ou_x", "also this", "oc_1", mem_sid="t-busy",
                                        image_handles=[{"kind": "file", "key": "f_1",
                                                        "name": "q3.pdf", "message_id": "om_9"}])
        assert out["steered"] is True
        (text, mid, _fresh), = agent_core._pending_steer["t-busy"]
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

    def stream(session, inputs, chat_id, message_id=None):
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
    (text, _mid, _fresh), = seen[1]
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

    def stream(session, inputs, chat_id, message_id=None):
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
    assert agent_core._pending_steer["t-r"] == [("replayed after consent", "m2", True)]
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
    stuck. The phase has its own line, its heartbeat, and its own step in the panel."""
    ln, card, _ = _recording_card()
    card.compacting(True)
    assert "整理较早的对话" in card._status_text()
    card.compacting(False)
    assert card._status_text().startswith("🤔")
    # The phase lasted long enough to be a step: rewind its start so it counts.
    card._phase, card._phase_since = "compact", card._phase_since - 2
    card.compacting(False)
    assert any(step.startswith("整理上下文") for step in card.summary()[1])


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
        base = agent_core._middleware()                 # images + steering
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


def test_agui_verifies_a_claimed_actor_rather_than_trusting_or_deriving_it():
    """A claimed actor is safe only because fetching that actor's vaulted token runs the
    ownership check, which asks Lark whose token it is and refuses a mismatch. Deriving the
    actor instead is not possible: customState participates in the vault lookup and is built
    from the actor. threadId still comes from the session, never from the request."""
    import agui, inspect
    src = inspect.getsource(agui.handle)
    # Session building is what fetches the token and runs the ownership check; an unverified
    # claim surfaces as auth_url rather than as somebody else's conversation.
    assert "aget_session" in src and 'session.get("auth_url")' in src
    assert 'thread_id = session["mem_sid"]' in src and '"threadId": thread_id' in src
    assert 'payload.get("threadId")' not in src


def test_the_actor_is_derived_from_the_vaulted_token_owner():
    """Lark is asked who the token belongs to; an unestablishable owner is refused rather
    than assumed."""
    import lark_3lo
    with mock.patch.object(lark_3lo, "_agentcore") as ac, \
         mock.patch.object(lark_3lo, "_token_owner", return_value="ou_alice"):
        ac.get_resource_oauth2_token.return_value = {"accessToken": "lark-tok"}
        assert lark_3lo.actor_from_workload_token("WAT") == ("actor", "lark:ou_alice")
    with mock.patch.object(lark_3lo, "_agentcore") as ac, \
         mock.patch.object(lark_3lo, "_token_owner", return_value=""):
        ac.get_resource_oauth2_token.return_value = {"accessToken": "lark-tok"}
        assert lark_3lo.actor_from_workload_token("WAT") == ("error", "")


def test_an_unconsented_caller_gets_needs_consent_not_a_guess():
    import lark_3lo
    with mock.patch.object(lark_3lo, "_agentcore") as ac:
        ac.get_resource_oauth2_token.return_value = {"authorizationUrl": "https://x"}
        assert lark_3lo.actor_from_workload_token("WAT") == ("needs_consent", "")


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
