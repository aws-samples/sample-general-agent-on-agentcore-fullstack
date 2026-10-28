"""Sessions and turns: checkpoints, repair, history, compaction, claims and the agent loop."""

import time
from unittest import mock

from langchain_core.messages import HumanMessage

import core
import middleware
from channels import web_files
from runtime import sessions, turns

from support import _ScriptedModel, _compactable, _dummy_tool, _fake_saver, _fresh_claims, _repair, _summary_update


# The conversation lives in the checkpointer, so the two ways this can go wrong are both
# silent: picking the fallback in a real deployment (history dies with the microVM, which
# reads as "the bot forgot everything") or mis-wiring the offload (a long conversation
# hits DynamoDB's 400 KB item cap mid-turn).

def test_checkpointer_falls_back_to_one_shared_in_process_saver():
    """No table configured — a session must still work, and the fallback has to be shared:
    a fresh InMemorySaver per session build would drop history every cache expiry."""
    with mock.patch.object(sessions, "_CHECKPOINT_TABLE", ""), \
         mock.patch.object(sessions, "_fallback_saver", None):
        first = sessions._checkpointer()
        second = sessions._checkpointer()
    from langgraph.checkpoint.memory import InMemorySaver
    assert isinstance(first, InMemorySaver)
    assert first is second


def test_checkpointer_uses_dynamodb_with_offload_when_configured():
    with mock.patch.object(sessions, "_CHECKPOINT_TABLE", "tbl"), \
         mock.patch.object(sessions, "_CHECKPOINT_BUCKET", "bkt"), \
         mock.patch("langgraph_checkpoint_aws.DynamoDBSaver") as saver:
        sessions._checkpointer()
    kwargs = saver.call_args.kwargs
    assert kwargs["table_name"] == "tbl"
    assert kwargs["ttl_seconds"] == sessions._CHECKPOINT_TTL
    # Compression keeps the spill rare; the spill keeps a long thread from failing hard.
    assert kwargs["enable_checkpoint_compression"] is True
    assert kwargs["s3_offload_config"]["bucket_name"] == "bkt"


def test_checkpointer_omits_offload_when_no_bucket():
    """Passing a config with an empty bucket name would fail at write time, deep inside
    the saver, instead of simply not offloading."""
    with mock.patch.object(sessions, "_CHECKPOINT_TABLE", "tbl"), \
         mock.patch.object(sessions, "_CHECKPOINT_BUCKET", ""), \
         mock.patch("langgraph_checkpoint_aws.DynamoDBSaver") as saver:
        sessions._checkpointer()
    assert saver.call_args.kwargs["s3_offload_config"] is None


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
    from langchain_core.messages import AIMessage as AI, HumanMessage
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


def test_the_thread_comes_from_the_router_never_from_a_client():
    """A thread is addressed by user alone, so a claimed id would read somebody else's
    conversation. The router owns rotation (/reset, /new) and is the only caller trusted to
    name one; a surface that cannot — the browser — inherits what it last named, which works
    because both share the router's runtime session id and land in one container."""
    sessions._thread_named.clear()
    derived = sessions._session_id_for("lark:ou_x")
    assert sessions.thread_for("lark:ou_x") == derived      # before anything is named
    rotated = derived + "-0123456789ab"
    sessions.remember_thread("lark:ou_x", rotated)
    assert sessions.thread_for("lark:ou_x") == rotated
    # An empty value must not erase what the router said, nor may another user's thread.
    sessions.remember_thread("lark:ou_x", "")
    sessions.remember_thread("lark:ou_x", sessions._session_id_for("lark:ou_other"))
    assert sessions.thread_for("lark:ou_x") == rotated
    # And it is per actor.
    assert sessions.thread_for("lark:ou_other") == sessions._session_id_for("lark:ou_other")


def test_claims_count_turns_and_tell_a_second_claimant_who_holds_the_thread():
    """/ping reports HealthyBusy while a turn runs, or AgentCore may reclaim the container
    mid-turn. A second release must not drive the count below zero, or it frees a claim
    another turn has taken meanwhile."""
    _fresh_claims()
    assert turns.busy() is False
    assert turns.try_claim("t-c", "web") is None
    assert turns.busy() is True and turns.try_claim("t-c", "lark") == "web"
    turns.release("t-c")
    turns.release("t-c")
    assert turns.thread_owner("t-c") is None and turns._in_flight == 0 and turns.busy() is False


def test_the_session_retry_hands_back_the_live_consent_link():
    """Each build mints a pending consent session and invalidates the previous one, so
    keeping the first attempt's auth_url gave the user a link the retry had already killed —
    CompleteResourceTokenAuth then answered "Invalid or expired session" every time."""
    calls = []

    def build(actor, email, mem_sid, wat="", own_claim=False):
        calls.append(1)
        return {"auth_url": f"https://consent/{len(calls)}", "mem_sid": mem_sid,
                "created": 0, "stack": None}

    with mock.patch.object(sessions, "_build_session", side_effect=build), \
         mock.patch.object(sessions, "_close_session"):
        s = sessions._get_session("lark:ou_x", "", "mem1", workload_token="wat")
    assert len(calls) == 2
    assert s["auth_url"] == "https://consent/2"   # the newest, not the first


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
    with mock.patch.dict(sessions._sessions, {"lark:u|mem1": cached}, clear=True), \
         mock.patch.object(sessions, "_build_session", fake_build), \
         mock.patch.object(sessions, "_close_session", lambda s: closed.append(s)), \
         mock.patch.object(time, "time", return_value=1e9 + 10):
        # Without fresh, the cached unauthorized session is returned (still inside TTL).
        s = sessions._get_session("lark:u", "", "mem1")
        assert s.get("auth_url") == "https://consent"
        assert built == []

        # With fresh, it is evicted, its MCP sessions closed, and a new one built.
        s = sessions._get_session("lark:u", "", "mem1", fresh=True)

    assert closed == [cached], "the stale MCP sessions must be closed, not leaked"
    assert built == ["mem1"]
    assert "auth_url" not in s


def test_history_stats_counts_a_turn_only_when_an_answer_was_delivered():
    """A tool-calling AIMessage is work in progress, not a turn — it is counted as a call."""
    from langchain_core.messages import AIMessage as AI, ToolMessage as TM
    msgs = [HumanMessage("q"), AI(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
            TM(content="r", tool_call_id="1"), AI(content="a")]
    with mock.patch.object(sessions, "_checkpointer", return_value=_fake_saver(msgs)):
        assert sessions.history_stats("lark:ou_x", "sess-1") == {
            "turns": 1, "toolCalls": 1, "messages": 3,
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
    with mock.patch.object(sessions, "_checkpointer", return_value=_fake_saver(msgs)):
        out = sessions.history_stats("lark:ou_x", "sess-1")
    assert (out["contextTokens"], out["cachedTokens"]) == (9147, 5049)
    # Summed over every call still in the checkpoint, for the thread's hit rate.
    assert (out["modelCalls"], out["inputTotal"], out["cachedTotal"]) == (2, 14115, 9881)


def test_compact_replaces_the_older_turns_with_a_summary_and_the_thread_goes_on():
    session, got, model = _compactable(n_turns=6, summary="Earlier: q0 to q2 were asked.")
    with got, model, mock.patch.object(sessions, "_COMPACT_KEEP", 4):
        out = sessions.compact("lark:ou_x", "t")
    msgs = turns._run(session["graph"].aget_state(session["config"])).values["messages"]
    assert out["compacted"] > 0 and out["kept"] == sum(1 for m in msgs if not middleware._is_summary(m))
    assert middleware._is_summary(msgs[0]) and "q0 to q2" in sessions._text_of(msgs[0])
    # Nothing pending: the next turn starts from the summary, as on a live thread.
    assert turns._run(session["graph"].aget_state(session["config"])).next == ()
    turns._run(session["graph"].ainvoke({"messages": [HumanMessage("next")]},
                                        config=session["config"]))


def test_compact_refuses_a_fallback_summary_rather_than_erase_the_history():
    """The failure the threshold path was fixed for — a history replaced by the fallback
    line — must not come back through the manual door."""
    session, got, model = _compactable(n_turns=6, summary=middleware._SUMMARY_FALLBACK)
    before = turns._run(session["graph"].aget_state(session["config"])).values["messages"]
    with got, model, mock.patch.object(sessions, "_COMPACT_KEEP", 4):
        out = sessions.compact("lark:ou_x", "t")
    after = turns._run(session["graph"].aget_state(session["config"])).values["messages"]
    assert out == {"compacted": 0, "reason": "failed"}
    assert [m.id for m in after] == [m.id for m in before]


def test_compact_leaves_a_short_thread_and_a_running_turn_alone():
    session, got, model = _compactable(n_turns=1, summary="s")
    with got, model, mock.patch.object(sessions, "_COMPACT_KEEP", 4):
        assert sessions.compact("lark:ou_x", "t")["reason"] == "short"
    with mock.patch.object(turns, "thread_busy", return_value=True):
        assert sessions.compact("lark:ou_x", "t") == {"compacted": 0, "reason": "busy"}


def test_the_tool_overview_keeps_assembly_order_and_names_the_missing_grants():
    session = {"tool_source": {"lark_whoami": "lark", "approval_list": "approval",
                               "run_code": "code"},
               "tool_idp": {"lark_whoami": "lark", "approval_list": "lark"},
               "consent": {"google": ""}}
    out = sessions.tools_overview(session)
    assert [t["name"] for t in out["tools"]] == ["lark_whoami", "approval_list", "run_code"]
    assert out["tools"][1] == {"name": "approval_list", "source": "approval", "idp": "lark"}
    assert out["ungranted"] == ["google"]
    assert sessions.tools_overview({"identity_error": "x"}) == {
        "tools": [], "identityError": "x"}


def test_transcript_rebuilds_what_a_reloaded_page_should_show():
    from langchain_core.messages import AIMessage as AI, ToolMessage as TM
    msgs = [HumanMessage("我有几份文档"),
            AI(content="", tool_calls=[{"name": "list_docs", "args": {"n": 5}, "id": "c1"}]),
            TM(content="5 份", tool_call_id="c1"),
            AI(content="你有 5 份。")]
    with mock.patch.object(sessions, "_checkpointer", return_value=_fake_saver(msgs)):
        out = sessions.transcript("lark:ou_x", "sess-1")
    assert [m["role"] for m in out["messages"]] == [
        "user", "toolCall", "toolResult", "assistant"]
    # The id is what pairs a call with its result when the page rebuilds the card.
    assert out["messages"][1]["id"] == out["messages"][2]["id"] == "c1"
    assert '"n": 5' in out["messages"][1]["args"]
    assert out["truncated"] is False


def test_transcript_reads_multimodal_content_without_crashing():
    """A turn that carried an image has a list of blocks, not a string."""
    with mock.patch.object(sessions, "_checkpointer", return_value=_fake_saver(
            [HumanMessage(content=[{"type": "text", "text": "看图"},
                                   {"type": "image", "source": {}}])])):
        out = sessions.transcript("lark:ou_x", "sess-1")
    assert out["messages"] == [{"role": "user", "content": "看图"}]


def test_transcript_caps_a_long_thread_and_says_so():
    msgs = [HumanMessage(f"q{i}") for i in range(sessions._TRANSCRIPT_LIMIT + 10)]
    with mock.patch.object(sessions, "_checkpointer", return_value=_fake_saver(msgs)):
        out = sessions.transcript("lark:ou_x", "sess-1")
    assert len(out["messages"]) == sessions._TRANSCRIPT_LIMIT
    assert out["truncated"] is True
    assert out["messages"][-1]["content"] == f"q{len(msgs) - 1}"   # the tail, not the head


def test_an_unreadable_checkpoint_degrades_status_and_the_transcript_instead_of_failing():
    """/status must still render, and a reloaded page is blank rather than broken."""
    broken = mock.Mock()
    broken.aget_tuple.side_effect = RuntimeError("no table")
    with mock.patch.object(sessions, "_checkpointer", return_value=broken):
        assert sessions.history_stats("lark:ou_x", "sess-1")["unavailable"] is True
        assert sessions.transcript("lark:ou_x", "sess-1") == {"messages": [], "unavailable": True}


def test_clear_history_deletes_the_thread_and_drops_the_cached_session():
    """A surviving cached session holds a compiled graph whose next turn would write on
    top of a thread the user was told is gone."""
    saver = _fake_saver()
    closed = []
    sessions._sessions["lark:ou_x|sess-1"] = {"stack": None, "created": 0}
    with mock.patch.object(sessions, "_checkpointer", return_value=saver), \
         mock.patch.object(sessions, "_close_session", closed.append):
        assert sessions.clear_history("lark:ou_x", "sess-1") == {"deleted": True}
    assert saver.deleted == "sess-1"
    assert "lark:ou_x|sess-1" not in sessions._sessions
    assert len(closed) == 1


def test_clear_history_reports_failure_rather_than_claiming_success():
    broken = mock.Mock()
    broken.adelete_thread.side_effect = RuntimeError("boom")
    with mock.patch.object(sessions, "_checkpointer", return_value=broken):
        r = sessions.clear_history("lark:ou_x", "sess-1")
    assert r["deleted"] is False and r["error"] == "RuntimeError"


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
        events = turns.stream_on_agent_loop(produce, on_done=done.set)
        async for _ in events:
            break            # the page went away after the first event
        assert not done.is_set(), "released while the run was still going"

    asyncio.run(read_one_and_leave())
    assert done.wait(5), "the run finished but never released"


def test_the_claim_holder_repairs_its_own_thread():
    """A claim is taken before the session is built, so the build-time repair sees the
    thread as busy — and skipping then left a buried dangling call failing every turn."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langchain_core.messages import AIMessage as _AI
    _fresh_claims()
    graph = core.create_agent(model=_ScriptedModel(turns=[{"text": ["ok"]}]), tools=[],
                              checkpointer=InMemorySaver(),
                              middleware=core._middleware()[:2])
    cfg = {"configurable": {"thread_id": "t-rep"}}
    turns._run(graph.aupdate_state(cfg, {"messages": [
        HumanMessage("q"),
        _AI(content="", tool_calls=[{"name": "t", "args": {}, "id": "lost", "type": "tool_call"}]),
        HumanMessage("asked again")]}, as_node="model"))
    turns.try_claim("t-rep", "lark")
    try:
        assert turns._run(sessions._arepair_interrupted_turn(graph, cfg)) == 0
        assert turns._run(sessions._arepair_interrupted_turn(graph, cfg, own_claim=True)) == 1
    finally:
        turns.release("t-rep")


def test_the_transcript_shows_the_summary_as_a_note_not_as_the_user():
    head = _summary_update("what happened so far")["messages"][1]
    tup = mock.Mock(checkpoint={"channel_values": {"messages": [head, HumanMessage("hi")]}})
    saver = mock.Mock(aget_tuple=mock.AsyncMock(return_value=tup))
    with mock.patch.object(sessions, "_checkpointer", return_value=saver):
        roles = [m["role"] for m in sessions.transcript("lark:ou_x")["messages"]]
    assert roles == ["summary", "user"]


def test_a_message_steers_into_its_own_surfaces_turn_and_only_that():
    turns._pending_steer.clear()
    assert turns.claim_or_steer("t-cs", "web", "web:tab-a", "first") is None
    try:
        assert turns.claim_or_steer("t-cs", "web", "web:tab-a", "also this", "m2") == "steered"
        # Another tab, a reload or another device would have the answer stream into this one.
        assert turns.claim_or_steer("t-cs", "web", "web:tab-b", "from b") == "web-other"
        assert turns.claim_or_steer("t-cs", "web", "", "no stream id") == "web-other"
        assert turns.claim_or_steer("t-cs", "lark", "oc_1", "from lark") == "web"
        assert [q[:2] for q in turns._pending_steer["t-cs"]] == [("also this", "m2")]
        # Taken as a continuation, the thread stays claimed; with nothing left it is freed.
        assert [q[0] for q in turns.take_steer_or_release("t-cs")] == ["also this"]
        assert turns.thread_owner("t-cs") == "web"
        assert turns.take_steer_or_release("t-cs") == []
        assert turns.thread_owner("t-cs") is None
    finally:
        turns.release("t-cs")


def test_the_history_shows_a_delivered_file_again_and_clear_removes_it():
    from langchain_core.messages import AIMessage as _AI, ToolMessage as _Tool
    from langgraph.checkpoint.memory import InMemorySaver
    saver = InMemorySaver()
    graph = core.create_agent(model=_ScriptedModel(turns=[{"text": ["ok"]}]),
                              tools=[_dummy_tool], checkpointer=saver)
    cfg = {"configurable": {"thread_id": "t-files"}}
    sent = {"file": {"key": "deliveries/o/u/chart.png", "name": "chart.png", "size": 3}}
    turns._run(graph.aupdate_state(cfg, {"messages": [
        HumanMessage("chart please", id="h1"),
        _AI(content="", id="a1", tool_calls=[{"name": "deliver_file", "args": {}, "id": "c1",
                                              "type": "tool_call"}]),
        _Tool(content="Sent chart.png to the page", tool_call_id="c1", artifact=sent),
        _AI(content="here it is", id="a2")]}, as_node="model"))
    with mock.patch.object(sessions, "_checkpointer", return_value=saver), \
         mock.patch.object(sessions, "_thread_config", return_value=cfg), \
         mock.patch.object(web_files, "url", return_value="https://fresh") as url, \
         mock.patch.object(web_files, "delete") as delete:
        files = [m for m in sessions.transcript("web:s-1")["messages"] if m["role"] == "file"]
        assert files == [{"role": "file", "name": "chart.png", "size": 3, "url": "https://fresh"}]
        url.assert_called_once_with("deliveries/o/u/chart.png", "chart.png", "web:s-1")
        assert sessions.clear_history("web:s-1")["deleted"] is True
        assert delete.call_args.args[0][0] == "deliveries/o/u/chart.png"
        assert delete.call_args.args[1] == "web:s-1"
