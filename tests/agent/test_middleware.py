"""Around each model call: steering, pictures, timestamps, summarisation."""

import asyncio
import time
from unittest import mock

import pytest
from langchain_core.messages import HumanMessage

import core
import middleware
from channels import images, lark
from runtime import sessions, turns

from support import FakeCard, _BLOCK, _ScriptedModel, _SummaryModel, _dummy_tool, _fresh_claims, _image_mw, _make_tool, _patch_notify, _register, _runtime, _summarising_session, _summary_update


def test_a_message_arriving_mid_turn_is_steered_not_refused():
    """Two turns on one thread each read the other's half-written checkpoint, which Bedrock
    rejects — the failure that made a live thread look broken. The second message joins the
    running turn instead, so nothing has to be re-sent."""
    turns._pending_steer.clear()
    turns.steer("t-1", "and also check the logs")
    turns.steer("t-1", "one more thing")
    mw = [m for m in core._middleware() if getattr(m, "name", "") == "steering"][0]
    # The runtime's real shape on LangGraph 1.2: the thread id lives on execution_info,
    # and there is no `config` attribute at all. A fake carrying `config` kept this test
    # green while steering was dead in production — every queued message was dropped.
    runtime = type("R", (), {"execution_info": type("E", (), {"thread_id": "t-1"})()})()
    assert not hasattr(runtime, "config")
    out = mw.before_model({"messages": []}, runtime)
    assert [m.content for m in out["messages"]] == ["and also check the logs", "one more thing"]
    # Drained, so the next model call in the same turn does not see them twice.
    assert mw.before_model({"messages": []}, runtime) is None
    # before_model is the one safe boundary: a toolResult must follow its toolUse directly,
    # so nothing may hook the tool boundary, where injecting would break that adjacency.
    assert not hasattr(middleware._SteeringMiddleware, "after_model")
    assert not hasattr(middleware._SteeringMiddleware, "wrap_tool_call")


def test_steering_is_per_thread():
    """One user's aside must not land in another user's turn."""
    turns._pending_steer.clear()
    turns.steer("t-a", "for A")
    mw = [m for m in core._middleware() if getattr(m, "name", "") == "steering"][0]
    assert mw.before_model({"messages": []}, _runtime("t-b")) is None
    assert [q[:3] for q in turns._pending_steer["t-a"]] == [("for A", "", False)]
    assert list(turns._pending_steer) == ["t-a"]


# The thread is permanent per user and DynamoDBSaver has no prune, so summarisation is the
# only bound. It must trigger on a threshold, never per turn: it rewrites the prefix, and
# rewriting the prefix every turn would make the prompt cache miss every turn.

def test_the_summariser_fires_on_a_token_threshold_and_stays_off_the_chat():
    """Per turn it would rewrite the cached prefix every turn. The library's 4,000-token
    trim cannot fit an agentic tail and came back empty; untagged, the summary streams like
    an answer; and its one-off call would only pay a cache-write premium."""
    mw = [m for m in core._middleware() if type(m).__name__ == "SummarizationMiddleware"]
    assert len(mw) == 1
    mw = mw[0]
    assert mw._trigger_clauses == [{"tokens": middleware._SUMMARIZE_AT_TOKENS}]
    assert middleware._SUMMARIZE_AT_TOKENS > 0
    assert mw.trim_tokens_to_summarize is None
    assert "nostream" in mw.model.tags and mw.model.metadata.get("emit-messages") is False
    assert mw.model.timeout == middleware._SUMMARY_TIMEOUT and mw.model.cache_ttl == ""


def test_a_message_steered_in_or_answered_later_keeps_the_time_it_arrived():
    """A message steered in mid-turn, or answered as a continuation after the turn, was said
    when it arrived — not when the turn began, and not when its answer started."""
    from langgraph.checkpoint.memory import InMemorySaver
    turns._pending_steer.clear()
    with mock.patch.object(middleware, "_SUMMARIZE_AT_TOKENS", 0):
        mw = core._middleware()
    tool = _make_tool("lookup")
    graph = core.create_agent(
        model=_ScriptedModel(turns=[{"text": ["looking"], "tool_calls": ["lookup"]},
                                    {"text": ["done"]}, {"text": ["later"]}]),
        tools=[tool], middleware=mw, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-steer-clock", "received_ms": 1000}}
    turns.steer("t-steer-clock", "also this", "h-steer")
    arrived = turns._pending_steer["t-steer-clock"][0][3]

    async def run(inputs):
        async for _ in graph.astream({"messages": inputs}, cfg, stream_mode="messages"):
            pass
    turns._run(run([HumanMessage("q", id="h-q")]))
    turns._run(run(lark._turn_input([("and later", "h-late", False, 5000)])))
    by_id = {m.id: m.additional_kwargs.get("ts")
             for m in turns._run(graph.aget_state(cfg)).values["messages"]}
    assert by_id["h-q"] == 1000 and by_id["h-steer"] == arrived and by_id["h-late"] == 5000


def test_a_message_whose_turn_stopped_before_a_reply_still_gets_its_time():
    """Stopped at a consent wall the model never returns, so the clock never stamps it."""
    from langgraph.checkpoint.memory import InMemorySaver
    with mock.patch.object(middleware, "_SUMMARIZE_AT_TOKENS", 0):
        mw = core._middleware()
    graph = core.create_agent(
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
    turns._run(full("q1", "h1"))
    turns._run(stopped("q2", "h2"))
    assert turns._run(sessions.astamp_unanswered(graph, cfg, 4242)) == 1
    assert turns._run(sessions.astamp_unanswered(graph, cfg, 9999)) == 0   # idempotent
    turns._run(full("q3", "h3"))
    msgs = turns._run(graph.aget_state(cfg)).values["messages"]
    by_id = {m.id: m.additional_kwargs.get("ts") for m in msgs if isinstance(m, HumanMessage)}
    assert by_id["h2"] == 4242 and [m.id for m in msgs if isinstance(m, HumanMessage)] == ["h1", "h2", "h3"]


def test_each_message_records_when_it_was_said():
    """A reloaded page shows the times Lark chat shows, so they live on the thread itself:
    the user's message and the reply each carry one, and the transcript hands them out."""
    from langgraph.checkpoint.memory import InMemorySaver
    with mock.patch.object(middleware, "_SUMMARIZE_AT_TOKENS", 0):
        mw = core._middleware()
    graph = core.create_agent(
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
    turns._run(first("q1", "h1"))
    turns._run(turn("q2", "h2"))
    after = int(time.time() * 1000)
    msgs = turns._run(graph.aget_state(cfg)).values["messages"]
    assert [type(m).__name__ for m in msgs] == ["HumanMessage", "AIMessage"] * 2
    stamps = [m.additional_kwargs.get("ts") for m in msgs]
    assert stamps[0] == before - 5000
    assert all(before <= t <= after for t in stamps[1:]) and stamps == sorted(stamps), stamps
    # Stamped in place: still one copy of each.
    assert [sessions._text_of(m) for m in msgs] == ["q1", "one", "q2", "two"]
    with mock.patch.object(sessions, "_checkpointer", return_value=graph.checkpointer), \
         mock.patch.object(sessions, "_thread_config", return_value=cfg):
        out = sessions.transcript("web:s-1")["messages"]
    assert [m.get("ts") for m in out] == stamps


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
    images.end_turn("t-px")


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
    images.end_turn("t-mid")
    assert images.pixels_for("t-mid") == {}


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
    graph = core.create_agent(model=_Recording(turns=[{"text": ["ok"]}]), tools=[],
                              checkpointer=InMemorySaver(),
                              middleware=core._middleware()[:2])
    with mock.patch.object(images, "blocks_for", return_value=[_BLOCK]):
        _, first = images.add_to_turn(
            "t-real", [{"kind": "image", "key": "img_a", "message_id": "om_1"}])
        marks, later = images.add_to_turn(
            "t-real", [{"kind": "image", "key": "img_b", "message_id": "om_2"}])
    turns.steer("t-real", "and this one\n" + marks, later)
    # Streamed, as a real turn is (_iter_deltas): that is the path that calls _astream.
    async def run():
        async for _ in graph.astream({"messages": [HumanMessage("look", id=first)]},
                                     config={"configurable": {"thread_id": "t-real"}},
                                     stream_mode="messages"):
            pass
    turns._run(run())
    humans = [m for m in seen_by_model[0] if isinstance(m, HumanMessage)]
    assert [m.id for m in humans] == [first, later]
    for m in humans:
        assert any(isinstance(b, dict) and b.get("type") == "image" for b in m.content), m.id
    images.end_turn("t-real")


def test_a_consent_replay_waits_for_the_continuation_on_a_rebuilt_session():
    """Injected into the running turn it would run on the session built before the grant
    and wall again, with the parked message already spent."""
    _fresh_claims()
    turns.steer("t-r", "aside", "m1")
    turns.steer("t-r", "replayed after consent", "m2", fresh=True)
    mw = [m for m in core._middleware() if getattr(m, "name", "") == "steering"][0]
    out = mw.before_model({"messages": []}, _runtime("t-r"))
    assert [m.content for m in out["messages"]] == ["aside"]
    assert [q[:3] for q in turns._pending_steer["t-r"]] == [("replayed after consent", "m2", True)]
    _fresh_claims()


def test_a_summary_inside_a_turn_tells_that_turns_card_start_and_end():
    """Through a compiled graph with the summariser forced to fire: the card hears exactly
    when a summary is being written, and nothing when the threshold check passes."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langgraph.checkpoint.memory import InMemorySaver
    summariser = middleware._summarization(FakeListChatModel(responses=["a summary"]),
                                           trigger=("messages", 3), keep=("messages", 1))
    graph = core.create_agent(model=_ScriptedModel(turns=[{"text": ["ok"]}]), tools=[],
                              checkpointer=InMemorySaver(), middleware=[summariser])
    cfg = {"configurable": {"thread_id": "t-sum"}}
    card = mock.Mock()
    turns._turn_cards["t-sum"] = card

    async def turn(text):
        async for _ in graph.astream({"messages": [HumanMessage(text)]}, config=cfg,
                                     stream_mode="messages"):
            pass
    try:
        turns._run(turn("one"))                 # below the trigger: no summary
        assert card.compacting.call_args_list == []
        turns._run(turn("two"))                 # past it: one summary
        assert [c.args for c in card.compacting.call_args_list] == [(True,), (False,)]
    finally:
        turns._turn_cards.pop("t-sum", None)


def test_a_broken_card_cannot_cost_the_summary():
    """The status signal runs inside the summariser; anything it raises would be read as a
    failed summary and skip the compaction."""
    turns._turn_cards["t-broken"] = object()          # no compacting() at all
    try:
        with mock.patch("langgraph.config.get_config",
                        return_value={"configurable": {"thread_id": "t-broken"}}):
            middleware._turn_status("compacting", True)     # must not raise
    finally:
        turns._turn_cards.pop("t-broken", None)


def test_compacting_outside_a_turn_is_quiet():
    """/compact summarises outside any graph run; that is expected, not worth a traceback."""
    with mock.patch.object(middleware.log, "warning") as warned:
        middleware._turn_status("compacting", True)
    assert not warned.called


@pytest.mark.parametrize("tagged", [True, False])
def test_a_tool_heavy_thread_is_really_summarised_and_none_of_it_reaches_the_card(tagged):
    """Untagged too: the stream filter alone must keep the summary model's tokens and the
    summary message off the card, in case a later change drops the tag."""
    summary = _SummaryModel(tags=["nostream"] if tagged else None)
    session, _ = _summarising_session(summary)
    card = FakeCard()
    with _patch_notify(card, []):
        text = lark._stream_to_chat(session, "next question", "oc_1")
    assert text == "the answer"
    assert all("SUMMARY" not in u and "Here is a summary" not in u for u in card.updates)
    head = session["graph"].get_state(session["config"]).values["messages"][0]
    assert middleware._is_summary(head)
    assert summary.calls >= 1
    assert "SUMMARY-OF-EARLIER-WORK" in head.content
    assert middleware._SUMMARY_FALLBACK not in head.content


@pytest.mark.parametrize("outcome", ["fallback", "raises"])
def test_a_failed_summary_keeps_the_history_instead_of_replacing_it(outcome):
    """Applying the fallback line is what deleted 114 messages; an exception would fail
    the turn and every one after it. Either way the thread is left as it was."""
    mw = middleware._summarization(_SummaryModel())
    base = type(mw).__mro__[1]
    if outcome == "fallback":
        patch = mock.patch.object(base, "abefore_model", mock.AsyncMock(
            return_value=_summary_update(middleware._SUMMARY_FALLBACK)))
    else:
        patch = mock.patch.object(base, "abefore_model",
                                  mock.AsyncMock(side_effect=RuntimeError("throttled")))
    with patch:
        assert asyncio.run(mw.abefore_model({"messages": []}, None)) is None


def test_a_real_summary_is_applied():
    mw = middleware._summarization(_SummaryModel())
    update = _summary_update("what happened so far")
    with mock.patch.object(type(mw).__mro__[1], "abefore_model",
                           mock.AsyncMock(return_value=update)):
        assert asyncio.run(mw.abefore_model({"messages": []}, None)) is update
