"""Lark (the card, chat_async), the web page (AG-UI), and the files and pictures each carries."""

import asyncio
import base64
import hashlib
import io
import json
import threading
import time
from unittest import mock

import pytest
from langchain_core.messages import HumanMessage

import core
import middleware
from channels import agui, delivery, images, lark, lark_card, web_files
from identity import grants
from runtime import sessions, turns


from support import AGENT, FakeCard, _CapturedResp, _InlineThread, _ScriptedModel, _SummaryModel, _deleted, _dummy_tool, _fresh_claims, _inserted, _load_card, _patch_notify, _recording_card, _session_for, _shown, _summarising_session, _thread_seq


def test_card_worker_writes_the_newest_text_in_order_and_never_loses_the_last():
    """Superseded states may be dropped — CardKit takes the full text every time — but the
    final text must be written, and the worker and close() share one strictly increasing
    sequence, since CardKit rejects an out-of-order one."""
    calls = []
    ln = _load_card(calls)
    card = ln.StreamingCard("oc_1")
    card.card_id = "c1"
    card.ok = True
    card._worker = threading.Thread(target=card._pump, daemon=True)
    card._worker.start()
    for t in ("a", "ab", "abc", "abcd"):
        assert card.update(t) is True
    card.close("abcd-final")
    contents = [c for _, c, _ in calls if c]
    assert "abcd-final" in contents[-2:], "the final text must always be written"
    assert all(c in ("a", "ab", "abc", "abcd", "abcd-final") for c in contents)
    seqs = [s for _, _, s in calls if s is not None]
    assert seqs == sorted(set(seqs)), f"sequence not strictly increasing: {seqs}"


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
    import time as _t
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


def test_the_status_animations_ship_with_the_image():
    """Read from agent/assets at run time; the Dockerfile has to copy them, and the image
    tag has to change with them or a new animation is never built."""
    import pathlib
    here = pathlib.Path(AGENT)
    for name in lark_card._ANIM_FILES.values():
        assert (here / "assets" / name).read_bytes()[:6] in (b"GIF89a", b"GIF87a"), name
    assert "COPY assets/" in (here / "Dockerfile").read_text()
    assert 'glob.glob("assets/*")' in (here.parent / "scripts" / "provision.sh").read_text()


def test_a_failed_animation_upload_is_not_retried_on_every_turn():
    import importlib
    importlib.reload(lark_card)
    with mock.patch.object(lark_card, "upload_image",
                           side_effect=lark_card.LarkError("no")) as up:
        assert lark_card._animation_keys() == {}
        assert lark_card._animation_keys() == {}
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
    import importlib
    importlib.reload(lark_card)           # earlier tests replace _call itself
    with mock.patch.object(lark_card.urllib.request, "urlopen",
                           side_effect=TimeoutError("slow")):
        assert lark_card._call("PUT", "https://x", {})["code"] == -1


def test_stream_hands_over_every_delta_cumulatively():
    """The loop no longer throttles: handing over text is a lock, not a round trip, so
    every delta goes over and the card's worker decides what to actually write. Each
    hand-over carries the full text so far — CardKit renders the appended tail, so a
    non-prefix would flash the wrong content."""
    card = FakeCard()
    deltas = ["a" * 30, "b" * 30, "c" * 30, "d" * 30]
    session = _session_for(deltas)
    with _patch_notify(card, []):
        text = lark._stream_to_chat(session, "msg", "oc_1")
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
        text = lark._stream_to_chat(session, "查待审批", "oc_1")
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
        text = lark._stream_to_chat(session, "今天天气", "oc_1")
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
        lark._stream_to_chat(session, "msg", "oc_1")
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
        turns.steer(thread, "不用看代码了")
        turns._steer_reply_to[thread] = "om_steered"
        return "found"
    steering = type("SteeringMiddleware", (AgentMiddleware,),
                    dict(middleware._SteeringMiddleware.__dict__))()
    graph = core.create_agent(
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
        with mock.patch.object(lark, "lark_card", notify):
            text = lark._stream_to_chat(session, "看一下仓库", "oc_group")
    finally:
        turns._pending_steer.pop(thread, None)
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
    turns.steer(thread, "B")
    turns._steer_reply_to[thread] = "om_b"
    steering = type("SteeringMiddleware", (AgentMiddleware,),
                    dict(middleware._SteeringMiddleware.__dict__))()
    graph = core.create_agent(model=_ScriptedModel(turns=[{"text": ["A 和 B 的答案"]}]),
                              tools=[_dummy_tool], middleware=[steering])
    session = {"graph": graph, "tool_texts": [],
               "config": {"configurable": {"thread_id": thread}}}
    made = []
    notify = mock.Mock(StreamingCard=lambda chat_id, **kw: made.append(FakeCard()) or made[-1],
                       send_text=lambda cid, t: True)
    with mock.patch.object(lark, "lark_card", notify):
        lark._stream_to_chat(session, "A", "oc_1")
    assert len(made) == 1 and made[0].closed_with == "A 和 B 的答案"


def test_a_failed_hand_over_still_delivers_the_rest():
    """The hand-over runs inside the stream: if it raises, the turn must go on as text
    rather than end in an apology with the graph still running."""
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.tools import StructuredTool
    thread = f"t{next(_thread_seq)}"

    def lookup() -> str:
        """Steers a message in mid-turn."""
        turns.steer(thread, "还有这个")
        return "found"
    steering = type("SteeringMiddleware", (AgentMiddleware,),
                    dict(middleware._SteeringMiddleware.__dict__))()
    graph = core.create_agent(
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
        with mock.patch.object(lark, "lark_card", notify):
            lark._stream_to_chat(session, "看一下", "oc_1")
    finally:
        turns._pending_steer.pop(thread, None)
    assert first.closed_with == "先查。" and sent == ["好了。"]


def test_stream_does_not_abort_when_authorized():
    """An authorized session has no auth_url, so tool calls proceed normally."""
    card = FakeCard()
    session = _session_for(["结果…"], tool_calls=["lark_list_my_docs"],
                           follow_up="共 3 个文件")     # no auth_url
    with _patch_notify(card, []):
        text = lark._stream_to_chat(session, "查文档", "oc_1")
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
        lark._stream_to_chat(session, "查文档", "oc_1")
    session["consent"] = {"lark": "https://consent"}   # the vault lookup that follows finds none
    assert any(grants._NEEDS_TOKEN_MARKER in t for _, t in session["tool_texts"])
    assert grants._hit_auth_wall(session) == "lark"


def test_stream_falls_back_to_text_when_card_cannot_open():
    """No cardkit:card:write scope → open() fails → the answer still arrives as text."""
    card = FakeCard(open_ok=False)
    sent = []
    session = _session_for(["hello ", "world"])
    with _patch_notify(card, sent):
        text = lark._stream_to_chat(session, "msg", "oc_1")
    assert text == "hello world"
    assert card.updates == []              # never tried to stream
    assert sent == ["hello world"]         # delivered as plain text instead


def test_stream_falls_back_when_an_update_fails_midway():
    """A card that dies mid-stream must still deliver the full answer via text."""
    card = FakeCard(update_ok=False)       # first update flips ok to False
    sent = []
    session = _session_for(["x" * 100, "y" * 100])
    with _patch_notify(card, sent):
        text = lark._stream_to_chat(session, "msg", "oc_1")
    assert text == "x"*100 + "y"*100
    assert sent == [text]                  # fell back after the failed update
    assert card.closed_with is None        # never reached a clean close


def test_an_attachment_steered_into_a_running_turn_is_not_dropped():
    """It used to be: the busy check came before the attachments were read, and steering
    carried text only, so a file sent while the bot was answering vanished silently."""
    turns._pending_steer.clear()
    turns._in_flight_threads.clear()
    turns._track(+1, "t-busy", "lark", "oc_1")
    try:
        with mock.patch.object(images, "blocks_for", return_value=[]), \
             mock.patch.object(lark, "lark_card"):
            # From another chat — a DM and a group share the thread — it is not steered: the
            # answer would appear in the chat the turn is answering.
            other = lark.chat_async("lark:ou_x", "elsewhere", "oc_2", mem_sid="t-busy")
            assert other["busy"] == "lark-other" and "t-busy" not in turns._pending_steer
            out = lark.chat_async("lark:ou_x", "also this", "oc_1", mem_sid="t-busy",
                                  image_handles=[{"kind": "file", "key": "f_1",
                                                  "name": "q3.pdf", "message_id": "om_9"}])
        assert out["steered"] is True
        (text, mid, _fresh, _arrived), = turns._pending_steer["t-busy"]
        assert "also this" in text and "save_attachment(1)" in text and "q3.pdf" in text
        assert images.handles_for("t-busy")[0]["key"] == "f_1"
    finally:
        turns._track(-1, "t-busy")
        images.end_turn("t-busy")
        turns._pending_steer.clear()


def test_a_lark_message_during_a_web_turn_is_told_not_steered():
    """Steered into a web turn, its answer would appear on the page while the user waits in
    Lark. Nothing is registered either, so the web turn's attachment numbering is untouched."""
    turns._pending_steer.clear()
    turns._in_flight_threads.clear()
    turns._track(+1, "t-web", "web")
    try:
        with mock.patch.object(lark, "lark_card") as notify:
            out = lark.chat_async("lark:ou_x", "hi", "oc_1", mem_sid="t-web",
                                  image_handles=[{"kind": "file", "key": "f_1",
                                                  "message_id": "om_9"}])
        assert out["busy"] == "web"
        assert "网页" in notify.send_text.call_args.args[1]
        assert turns._pending_steer == {} and images.handles_for("t-web") == []
    finally:
        turns._track(-1, "t-web")


def test_a_turn_releases_its_thread_and_attachments_on_every_way_out():
    """Left claimed, the thread reads as busy for good and every later message is steered
    into a turn that no longer exists."""
    handles = [{"kind": "file", "key": "f_1", "name": "a.pdf", "message_id": "om_1"}]
    for case in ("identity", "raises", "ok"):
        _fresh_claims()
        if case == "identity":
            got = mock.patch.object(sessions, "_get_session",
                                    return_value={"identity_error": "nope"})
        elif case == "raises":
            got = mock.patch.object(sessions, "_get_session", side_effect=RuntimeError("cold"))
        else:
            got = mock.patch.object(sessions, "_get_session", return_value=_session_for([]))
        with got, mock.patch.object(lark, "lark_card"), \
             mock.patch.object(lark, "_stream_to_chat", return_value="ok"), \
             mock.patch.object(threading, "Thread", _InlineThread):
            try:
                lark.chat_async("lark:ou_x", "hi", "oc_1", mem_sid="t-out",
                                image_handles=handles)
            except RuntimeError:
                assert case == "raises"
        assert turns.thread_owner("t-out") is None, case
        assert images.handles_for("t-out") == [], case


def test_a_message_steered_after_the_last_model_call_is_answered_not_dropped():
    """That window is the whole streamed answer, which is when people add "also this". The
    turn keeps its claim and answers the leftover as a continuation, attachments intact."""
    _fresh_claims()
    seen = []

    def stream(session, inputs, chat_id, message_id=None, received_ms=None):
        seen.append(inputs)
        if len(seen) == 1:       # arrives while the first answer is streaming
            with mock.patch.object(images, "blocks_for", return_value=[]):
                lark.chat_async("lark:ou_x", "also this", "oc_1", mem_sid="t-late",
                                image_handles=[{"kind": "file", "key": "f_9",
                                                "name": "late.pdf", "message_id": "om_9"}])
            assert images.handles_for("t-late"), "still registered for the continuation"
        return "ok"

    with mock.patch.object(sessions, "_get_session", return_value=_session_for([])), \
         mock.patch.object(lark, "lark_card"), \
         mock.patch.object(lark, "_stream_to_chat", side_effect=stream), \
         mock.patch.object(threading, "Thread", _InlineThread):
        lark.chat_async("lark:ou_x", "first", "oc_1", mem_sid="t-late")
    assert len(seen) == 2
    (text, _mid, _fresh, _arrived), = seen[1]
    assert "also this" in text and "save_attachment(1)" in text
    assert turns.thread_owner("t-late") is None
    assert images.handles_for("t-late") == []


def test_across_surfaces_a_message_is_told_not_steered():
    """Steered across surfaces, the answer appears in the other window — and a sync caller
    would be promised an answer it never receives."""
    for owner, arrives in (("sync", "lark"), ("web", "lark"), ("lark", "sync"), ("web", "sync")):
        _fresh_claims()
        turns.try_claim("t-x", owner)
        with mock.patch.object(lark, "lark_card") as notify:
            if arrives == "lark":
                out = lark.chat_async("lark:ou_x", "hi", "oc_1", mem_sid="t-x")
                assert out["busy"] == owner and notify.send_text.called
            else:
                out = lark.chat_result("lark:ou_x", "hi", mem_sid="t-x")
                assert out["busy"] == owner and "steered" not in out
        assert turns._pending_steer == {}, (owner, arrives)
        turns.release("t-x")


def test_a_malformed_page_request_does_not_leave_the_thread_claimed():
    """Claimed before the run input is built, so a payload that fails validation must
    release it — left claimed, Lark answers "正在网页上回答" for good."""
    _fresh_claims()
    session = {"mem_sid": "t-bad", "graph": object(), "config": {"configurable": {}}}

    async def go():
        with mock.patch.object(sessions, "aget_session",
                               mock.AsyncMock(return_value=session)):
            await agui.handle(mock.Mock(), {"actorId": "lark:ou_x", "messages": "not a list"}, "")

    try:
        asyncio.run(go())
    except Exception:  # noqa: BLE001 — the validation error itself is not under test
        pass
    assert turns.thread_owner("t-bad") is None and turns._in_flight == 0


def test_a_steer_queued_while_a_turn_failed_to_start_is_reported_not_carried_over():
    """Acknowledged as steered, so dropping it silently loses it; carried over, it would be
    injected into the next turn with its pixels gone and its numbers pointing elsewhere."""
    _fresh_claims()

    def cold_build(*a, **k):
        turns.steer("t-fail", "meanwhile", "msg-x")     # arrives during the build
        raise RuntimeError("build failed")

    with mock.patch.object(sessions, "_get_session", side_effect=cold_build), \
         mock.patch.object(lark, "lark_card") as notify:
        try:
            lark.chat_async("lark:ou_x", "first", "oc_1", mem_sid="t-fail")
        except RuntimeError:
            pass
    assert turns._pending_steer == {}
    assert any("1 条消息没来得及处理" in c.args[1] for c in notify.send_text.call_args_list)
    assert turns.thread_owner("t-fail") is None


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
            turns.steer("t-cont", "after consent", "m-r", fresh=True)
        return "ok"

    with mock.patch.object(sessions, "_get_session", side_effect=get_session), \
         mock.patch.object(sessions, "_drop_sessions"), \
         mock.patch.object(lark, "lark_card") as notify, \
         mock.patch.object(lark, "_stream_to_chat", side_effect=stream), \
         mock.patch.object(threading, "Thread", _InlineThread):
        lark.chat_async("lark:ou_x", "first", "oc_1", mem_sid="t-cont")
    sent = [c.args[1] for c in notify.send_text.call_args_list]
    assert any("1 条消息没来得及处理" in m and "RuntimeError" in m for m in sent), sent
    assert turns.thread_owner("t-cont") is None


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


def test_a_turn_unregisters_its_card_even_when_it_fails():
    card = mock.Mock(open=mock.Mock(return_value=True))
    session = {"graph": object(), "config": {"configurable": {"thread_id": "t-fail-card"}},
               "tool_texts": []}
    with mock.patch.object(lark_card, "StreamingCard", return_value=card), \
         mock.patch.object(lark, "_iter_deltas", side_effect=RuntimeError("boom")):
        try:
            lark._stream_to_chat(session, "hi", "oc_1")
        except RuntimeError:
            pass
    assert "t-fail-card" not in turns._turn_cards


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
    with mock.patch.object(sessions, "_get_session", return_value=session):
        out = lark.chat_result("lark:ou_x", "today's question", mem_sid="t-sync")
    assert session["tool_texts"] == [], "an old tool result leaked into this turn"
    assert out == {"reply": "fresh answer", "needs_auth": False}


def test_a_steered_message_is_not_echoed_onto_the_card():
    """Steering writes a HumanMessage from a node, which the stream replays."""
    session, thread = _summarising_session(_SummaryModel(tags=["nostream"]))
    turns.steer(thread, "STEERED-IN")
    card = FakeCard()
    with _patch_notify(card, []):
        text = lark._stream_to_chat(session, "next question", "oc_1")
    assert "STEERED-IN" not in text


# The browser talks to the Runtime directly here, so the one invariant that matters is that
# nothing it sends can name a user: threadId decides whose conversation and whose files.

def test_agui_requests_are_told_apart_from_the_router_protocol():
    assert agui.is_agui_request({"threadId": "t", "messages": []}) is True
    assert agui.is_agui_request({"runId": "r"}) is True
    # The router always sends an action, even alongside a thread id.
    assert agui.is_agui_request({"action": "chat", "memorySessionId": "m"}) is False
    assert agui.is_agui_request({}) is False


def test_the_page_is_sent_only_what_it_renders():
    """Snapshots repeat the whole thread and raw events the whole LangGraph payload: a few
    large tool results made one turn's stream 12 MB, which the browser dropped."""
    import json as _j
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
    adapter = LangGraphAgent(name="t", graph=graph, config=turns._run_config(
        session, delivery.NoDelivery("test")))

    async def events():
        return [e async for e in adapter.run(run)]
    kinds = [getattr(e, "type", None) for e in asyncio.run(events())]
    assert "RUN_ERROR" not in [str(k).split(".")[-1] for k in kinds], kinds


def test_a_web_turn_stops_at_a_tool_that_needs_a_grant():
    """As a Lark turn does: the model must not go on to narrate a refusal that is certain,
    and the page gets the consent prompt and a finished run."""
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
    with mock.patch.object(sessions, "aget_session", mock.AsyncMock(return_value=session)), \
         mock.patch("ag_ui_langgraph.LangGraphAgent", Run), \
         mock.patch.object(agui, "_sse", lambda e: f"data: {e.type}\n\n".encode()), \
         mock.patch.object(agui.web, "StreamResponse", Resp):
        resp = asyncio.run(agui.handle(None, {"actorId": "web:s-1", "runId": "r-1",
                                              "memorySessionId": "sess-x", "messages": []}, ""))
    frames = [f[len("data: "):] for f in resp.out.decode().split("\n\n") if f]
    assert frames[:2] == ["RUN_STARTED", "TEXT_MESSAGE_CONTENT"]
    assert json.loads(frames[2]) == {"type": "CUSTOM", "name": "consent", "value": {"idp": "lark"}}
    assert json.loads(frames[3])["type"] == "RUN_FINISHED" and len(frames) == 4
    assert reached == [] and turns.try_claim("sess-x", "web") is None
    turns.release("sess-x")


def test_a_file_the_run_sends_reaches_the_page_on_that_runs_stream():
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
    with mock.patch.object(sessions, "aget_session", mock.AsyncMock(return_value=session)), \
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
    turns.release("sess-f")


def test_a_message_sent_from_the_page_mid_turn_joins_it_and_is_answered_below():
    """As in Lark chat: what the page sends while its turn runs is not refused as busy. Too
    late for the running model call, it is answered as a continuation on the same stream,
    marked so the page starts the reply under that message."""
    from types import SimpleNamespace as E
    turns._pending_steer.clear()
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
    with mock.patch.object(sessions, "aget_session", mock.AsyncMock(return_value=session)), \
         mock.patch("ag_ui_langgraph.LangGraphAgent", Run), \
         mock.patch.object(agui, "_sse", lambda e: f"data: {e.type}\n\n".encode()), \
         mock.patch.object(agui.web, "StreamResponse", lambda **kw: resp):
        asyncio.run(agui.handle(None, {**page, "runId": "r-1", "messages": [
            {"id": "m1", "role": "user", "content": "first"}]}, ""))
    frames = resp.frames()
    assert inputs == [["first"], ["also this"]]
    assert frames[:2] == ["RUN_STARTED", "RUN_FINISHED"] and frames[-1] == "RUN_FINISHED"
    assert json.loads(frames[2])["name"] == "steered" and frames[3] == "RUN_STARTED"
    assert turns.thread_owner("sess-st") is None and "sess-st" not in turns._turn_cards


def test_a_message_steered_into_a_turn_that_stops_for_consent_is_reported_not_lost():
    from types import SimpleNamespace as E
    turns._pending_steer.clear()

    class Run:
        def __init__(self, **kw): pass

        async def run(self, _input):
            yield E(type="RUN_STARTED")
            assert turns.claim_or_steer("sess-w", "web", "web:tab-a", "also", "m2") == "steered"
            yield E(type="TOOL_CALL_START", tool_call_id="t1", tool_call_name="lark_docs")

    session = {"mem_sid": "sess-w", "graph": None, "config": {"configurable": {}},
               "tool_idp": {"lark_docs": "lark"}, "consent": {"lark": ""}}
    resp = _CapturedResp()
    with mock.patch.object(sessions, "aget_session", mock.AsyncMock(return_value=session)), \
         mock.patch("ag_ui_langgraph.LangGraphAgent", Run), \
         mock.patch.object(agui, "_sse", lambda e: f"data: {e.type}\n\n".encode()), \
         mock.patch.object(agui.web, "StreamResponse", lambda **kw: resp):
        asyncio.run(agui.handle(None, {"actorId": "web:s-1", "memorySessionId": "sess-w",
                                       "forwardedProps": {"streamId": "tab-a"}, "runId": "r",
                                       "messages": []}, ""))
    marks = [json.loads(f) for f in resp.frames() if f.startswith("{")]
    assert {"type": "CUSTOM", "name": "unanswered", "value": {"count": 1}} in marks
    assert turns.thread_owner("sess-w") is None


def test_a_web_run_never_releases_a_claim_it_no_longer_holds():
    """The run releases in the same step as finding nothing left; releasing again when the
    producer ends could free a turn that claimed the thread in between."""
    state = {"released": True}
    turns._track(+1, "sess-r", "lark", "oc_1")      # someone else's turn, meanwhile
    try:
        assert agui._end("sess-r", agui._PageMarks([]), state) == []
        assert turns.thread_owner("sess-r") == "lark"
    finally:
        turns.release("sess-r")


def test_a_delivered_file_is_stored_under_its_owner_and_only_media_is_shown_inline():
    s3 = mock.Mock()
    s3.generate_presigned_url.side_effect = lambda op, ExpiresIn, Params: f"https://s/{Params['Key']}"
    with mock.patch.object(web_files, "_BUCKET", "b"), mock.patch.object(web_files, "_s3", return_value=s3):
        key = web_files.put("web:s-1", "../chart.png", b"x")
        owner = hashlib.sha256(b"web:s-1").hexdigest()[:32]
        assert key.startswith(f"deliveries/{owner}/") and key.endswith("/chart.png")
        put = s3.put_object.call_args.kwargs
        assert put["ContentType"] == "image/png" and put["ContentDisposition"].startswith("inline")
        web_files.put("web:s-1", "clip.mp4", b"x")
        mp4 = s3.put_object.call_args.kwargs
        assert mp4["ContentType"] == "video/mp4" and mp4["ContentDisposition"].startswith("inline")
        web_files.put("web:s-1", "page.svg", b"<svg/>")
        svg = s3.put_object.call_args.kwargs
        assert svg["ContentType"] == "application/octet-stream"
        assert svg["ContentDisposition"].startswith("attachment")
        assert web_files.url(key, "chart.png", "web:s-1") == f"https://s/{key}"
        # Only this actor's own deliveries are signed or deleted.
        assert web_files.url(key, "chart.png", "web:someone-else") == ""
        assert web_files.url("state/x", "x.png", "web:s-1") == ""
        assert web_files.delete(["state/x", key, key], "web:s-1") == 1
        assert web_files.delete([key], "web:someone-else") == 0


def test_a_picture_from_the_page_is_seen_on_its_turn_and_kept_as_a_placeholder():
    """As a Lark picture is: inlined into the checkpoint it would be re-billed on every turn."""
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
            seen["pixels"] = images.pixels_for("sess-img")
            seen["handles"] = images.handles_for("sess-img")
            yield E(type="RUN_STARTED")
            yield E(type="RUN_FINISHED")

    session = {"mem_sid": "sess-img", "graph": None, "config": {"configurable": {}},
               "tool_idp": {}, "consent": {}}
    with mock.patch.object(sessions, "aget_session", mock.AsyncMock(return_value=session)), \
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
    assert images.pixels_for("sess-img") == {}         # gone with the turn


def test_a_picture_the_model_cannot_read_is_kept_but_not_shown_to_it():
    """Labelled as a jpeg, an unreadable picture would make Bedrock refuse the whole turn."""
    text, handles, blocks = images.from_page([
        {"type": "text", "text": "look"},
        {"type": "image", "source": {"type": "data", "value": base64.b64encode(b"not an image").decode(),
                                     "mimeType": "image/heic"}}])
    assert text == "look" and blocks == [] and handles[0]["name"] == "image-1.heic"
    marks, _ = images.register("t-heic", handles, blocks)
    assert "not a picture the model can read" in marks and "save_attachment" in marks
    images.end_turn("t-heic")


def test_a_web_run_stamps_an_unanswered_message_only_when_it_ended_without_a_reply():
    """Stopped at a wall, or failed: stamped with that run's own message's arrival, while the
    claim is held. A run that replied is left to the clock and costs no extra read."""
    from types import SimpleNamespace as E
    turns._pending_steer.clear()
    stamped, owners = [], []

    async def fake_stamp(graph, config, at_ms):
        stamped.append(at_ms)
        owners.append(turns.thread_owner("sess-sm"))
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
        with mock.patch.object(sessions, "aget_session", mock.AsyncMock(return_value=session)), \
             mock.patch.object(sessions, "astamp_unanswered", fake_stamp), \
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
        turns.steer("sess-sm", "also", "m2")
    at, held = run_with([[E(type="RUN_STARTED"), steer, E(type="RUN_FINISHED")],
                         [E(type="RUN_STARTED"),
                          E(type="TOOL_CALL_START", tool_call_id="t", tool_call_name="lark_docs")]])
    assert held == ["web"] and len(at) == 1 and at[0] >= sent["ms"]
    assert turns.thread_owner("sess-sm") is None


def test_a_refused_tool_on_the_page_ends_the_run_with_a_consent_prompt():
    """The page gets the system to authorise, not a link: it mints one through /auth when the
    user clicks, with the token it will complete with."""
    session = {"tool_texts": [("google_whoami", "no user token (authorize first)")],
               "tool_idp": {"google_whoami": "google"}, "consent": {"google": ""}}
    event = json.loads(agui._consent(session).decode()[len("data: "):])
    assert event == {"type": "CUSTOM", "name": "consent", "value": {"idp": "google"}}
    assert agui._consent({"tool_texts": [], "tool_idp": {}, "consent": {}}) == b""


def test_an_id_from_a_request_cannot_address_another_endpoint():
    """`messages/{id}?` is the recall endpoint, so a reaction id carrying `?` would have
    recalled any message the bot sent; a key carrying `../` would read another chat's."""
    import importlib
    importlib.reload(lark_card)
    calls = []
    with mock.patch.object(lark_card, "_tenant_token", return_value="t"), \
         mock.patch.object(lark_card, "_call", side_effect=lambda *a, **k: calls.append(a) or {"code": 0}):
        assert lark_card.remove_reaction("om_victim?", "x") is False
        assert lark_card.remove_reaction("om_1", "../chats/oc_x") is False
        assert lark_card.remove_reaction("om_1", "ZCaCIjUBVVW-rm5L_3Z=") is True
    assert len(calls) == 1
    with mock.patch.object(lark_card, "_tenant_token", return_value="t"):
        with pytest.raises(lark_card.LarkError):
            lark_card.fetch_message_resource("om_mine", "../../om_theirs/resources/img_x")
    assert lark_card.chat_of("om_1/../om_2") == ""


def test_membership_is_read_from_lark_and_cached_only_when_true():
    import importlib
    importlib.reload(lark_card)
    pages = [{"code": 0, "data": {"items": [{"member_id": "ou_x"}], "has_more": True,
                                  "page_token": "p2"}},
             {"code": 0, "data": {"items": [{"member_id": "ou_alice"}], "has_more": False}}]
    with mock.patch.object(lark_card, "_get", side_effect=pages) as get:
        assert lark_card.in_chat("oc_1", "ou_alice") is True
        assert lark_card.in_chat("oc_1", "ou_alice") is True      # cached
    assert get.call_count == 2
    with mock.patch.object(lark_card, "_get", return_value={"code": 1, "msg": "no"}):
        assert lark_card.in_chat("oc_2", "ou_alice") is False


def test_stream_walls_a_google_tool_with_the_google_card():
    card = FakeCard()
    session = _session_for(["查一下…"], tool_calls=["google_whoami"], follow_up="没有权限",
                           consent={"google": ""}, tool_idp={"google_whoami": "google"})
    with _patch_notify(card, []):
        text = lark._stream_to_chat(session, "我的 Google 账号", "oc_1")
    assert session.get("walled_tool") == "google_whoami"
    assert "Google" in text and "Lark" not in text
