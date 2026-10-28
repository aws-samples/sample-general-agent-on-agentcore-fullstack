"""The browser tools: one session per conversation, started on first use and kept across session
rebuilds, never started by the SDK behind our back, and screenshots shown for one step only."""

import asyncio
import time
from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import core
from channels import images
from runtime import turns
from tools import browser


class _Client:
    """BrowserClient as tools/browser.py uses it. Unlike the real one, it refuses a CDP header
    request without a session — the real one would quietly start a session nobody owns."""
    made, started, stopped = [], [], []

    def __init__(self, region):
        self.session_id = None
        _Client.made.append(self)

    def start(self, session_timeout_seconds=None, viewport=None):
        self.session_id = f"s{len(_Client.started) + 1}"
        _Client.started.append((self.session_id, session_timeout_seconds, viewport))
        return self.session_id

    def generate_ws_headers(self):
        assert self.session_id, "asked for CDP headers with no session"
        return f"wss://browser/{self.session_id}", {"Authorization": "sig"}

    def stop(self):
        _Client.stopped.append(self.session_id)
        self.session_id = None


class _Locator:
    def __init__(self, page, ref):
        self.page, self.ref = page, ref

    async def click(self, timeout=None):
        self.page.done.append(("click", self.ref))

    async def fill(self, text, timeout=None):
        self.page.done.append(("fill", self.ref, text))

    async def press(self, key, timeout=None):
        self.page.done.append(("press", self.ref, key))


class _Page:
    def __init__(self, snapshot='- link "Docs" [ref=e1]'):
        self.url, self.snapshot, self.done, self.closed = "about:blank", snapshot, [], False

    def is_closed(self):
        return self.closed

    async def goto(self, url, wait_until=None, timeout=None):
        self.url = url
        return mock.Mock(status=200)

    async def wait_for_load_state(self, state, timeout=None):
        pass

    async def title(self):
        return "A page"

    async def aria_snapshot(self, mode=None, timeout=None):
        assert mode == "ai"          # the mode that marks refs
        return self.snapshot

    def get_by_ref(self, ref):
        return _Locator(self, ref)

    async def screenshot(self, type=None, quality=None):
        return b"\xff\xd8jpeg"


class _Remote:
    def __init__(self, page):
        self.connected = True
        self.contexts = [mock.Mock(pages=[page])]

    def is_connected(self):
        return self.connected

    async def close(self):
        self.connected = False


class _Driver:
    """Playwright's chromium.connect_over_cdp; `dead` names sessions that no longer exist."""

    def __init__(self):
        self.dead, self.remotes, self.page = set(), [], _Page()
        self.chromium = self

    async def connect_over_cdp(self, ws, headers=None):
        if ws.rsplit("/", 1)[-1] in self.dead:
            raise ConnectionError("session not found")
        self.remotes.append(_Remote(self.page))
        return self.remotes[-1]


@pytest.fixture
def fake():
    _Client.made, _Client.started, _Client.stopped = [], [], []
    driver = _Driver()

    async def playwright():
        return driver
    with mock.patch.object(browser, "_ENABLED", True), \
         mock.patch.object(browser, "_playwright", playwright), \
         mock.patch("bedrock_agentcore.tools.browser_client.BrowserClient", _Client):
        yield driver


def _tools(b):
    return {t.name: t for t in browser.tools_for(b)}


def _call(tool, args=None, call_id="c1", thread="t-1"):
    """Through the agent loop, as a turn runs it: the browser's lock and page live there."""
    return turns._run(tool.ainvoke({"name": tool.name, "args": args or {}, "id": call_id,
                                    "type": "tool_call"},
                                   config={"configurable": {"thread_id": thread}}))


def test_a_browser_starts_on_first_use_and_is_reused(fake):
    """Starting costs a session (billed) and seconds; a conversation keeps one."""
    tools = _tools(browser._Browser("web:a"))
    assert _Client.started == []                           # nothing until a tool runs
    out = _call(tools["browser_open"], {"url": "https://example.com"})
    assert "A page — https://example.com (HTTP 200)" in out.content and "[ref=e1]" in out.content
    _call(tools["browser_snapshot"])
    assert [s[0] for s in _Client.started] == ["s1"]
    assert _Client.started[0][1:] == (browser._SESSION_TIMEOUT, browser._VIEWPORT)


def test_a_dropped_connection_reconnects_and_an_expired_session_is_replaced(fake):
    """A dropped socket keeps the session, so the page survives; an expired one cannot, and the
    model is told its page is gone rather than left clicking refs of a page that no longer is."""
    b = browser._Browser("web:a")
    tools = _tools(b)
    _call(tools["browser_open"], {"url": "https://example.com"})
    fake.remotes[-1].connected = False
    assert "restarted" not in _call(tools["browser_snapshot"]).content
    assert len(_Client.started) == 1 and len(fake.remotes) == 2       # same session, new socket
    fake.remotes[-1].connected = False
    fake.dead.add("s1")
    out = _call(tools["browser_click"], {"ref": "e1"})
    assert out.content.startswith(browser._RESTARTED) and "Open the page again" in out.content
    assert _Client.stopped == ["s1"] and [s[0] for s in _Client.started] == ["s1", "s2"]
    assert fake.page.done == []                                         # the stale ref was not clicked


def test_a_closed_browser_stops_its_session_and_is_never_restarted(fake):
    b = browser._Browser("web:a")
    tools = _tools(b)
    _call(tools["browser_open"], {"url": "https://example.com"})
    turns._run(b.aclose())
    assert _Client.stopped == ["s1"]
    out = _call(tools["browser_snapshot"])
    assert "has been closed" in out.content and len(_Client.started) == 1


def test_one_turns_browser_calls_run_one_at_a_time(fake):
    """A turn's tool calls run concurrently, and two of them on one page would interleave."""
    b = browser._Browser("web:a")
    tools = _tools(b)
    inside, overlap = [0], []

    async def slow_snapshot(mode=None, timeout=None):
        inside[0] += 1
        overlap.append(inside[0])
        await asyncio.sleep(0.05)
        inside[0] -= 1
        return "- text"
    fake.page.aria_snapshot = slow_snapshot

    async def both():
        return await asyncio.gather(*(tools["browser_snapshot"].ainvoke(
            {"name": "browser_snapshot", "args": {}, "id": f"c{i}", "type": "tool_call"})
            for i in range(3)))
    turns._run(both())
    assert max(overlap) == 1


def test_only_web_addresses_are_opened(fake):
    out = _call(_tools(browser._Browser("web:a"))["browser_open"], {"url": "file:///etc/passwd"})
    assert "http(s)" in out.content and _Client.started == []


def test_a_long_page_is_cut_and_says_so(fake):
    fake.page.snapshot = "x" * (browser._MAX_TEXT + 500)
    out = _call(_tools(browser._Browser("web:a"))["browser_snapshot"]).content
    assert out.endswith("(500 more characters not shown)") and len(out) < browser._MAX_TEXT + 200


def test_typing_and_submitting_act_on_the_ref(fake):
    tools = _tools(browser._Browser("web:a"))
    _call(tools["browser_open"], {"url": "https://example.com"})
    out = _call(tools["browser_type"], {"ref": "e7", "text": "otters", "submit": True})
    assert fake.page.done == [("fill", "e7", "otters"), ("press", "e7", "Enter")]
    assert out.content.startswith("Typed into e7.")


def test_while_the_user_holds_the_browser_the_tools_leave_it_alone(fake):
    b = browser._Browser("web:a")
    b.held_by_user = True
    out = _call(_tools(b)["browser_open"], {"url": "https://example.com"})
    assert "The user has the browser" in out.content and _Client.started == []


def test_a_screenshot_is_shown_for_its_step_and_never_stored(fake):
    """An image costs tokens on every request that carries it, and the thread is replayed each
    turn: the result keeps text, the pixels ride only on this turn's requests."""
    tools = _tools(browser._Browser("web:a"))
    try:
        out = _call(tools["browser_screenshot"], call_id="shot-1", thread="t-shot")
        assert out.content == "Screenshot of about:blank, shown to you at this step only."
        assert "tool:shot-1" in images.pixels_for("t-shot")
        stored = ToolMessage(content=out.content, tool_call_id="shot-1")
        call = AIMessage(content="", tool_calls=[{"name": "browser_screenshot", "args": {}, "id": "shot-1"}])
        mw = [m for m in core._middleware() if getattr(m, "name", "") == "images"][0]
        runtime = type("R", (), {"execution_info": type("E", (), {"thread_id": "t-shot"})()})()
        request = type("Q", (), {"messages": [HumanMessage("look"), call, stored], "runtime": runtime})()
        sent = mw._attach(request).messages[-1]
        assert sent.tool_call_id == "shot-1" and [b["type"] for b in sent.content] == ["text", "image"]
        assert stored.content == out.content                       # the thread keeps the text
    finally:
        images.end_turn("t-shot")


def test_a_screenshot_that_cannot_be_sent_is_not_said_to_be_in_the_workspace(fake):
    """Seen on v07: the code tools' sentence pointed the model at a workspace file that a
    screenshot never had, and the model repeated it to the user."""
    try:
        out = _call(_tools(browser._Browser("web:a"))["browser_screenshot"],
                    {"send_to_user": True}, call_id="shot-2", thread="t-send")
        assert "was not sent" in out.content and "workspace" not in out.content
    finally:
        images.end_turn("t-send")


def test_only_the_newest_screenshots_ride_on_a_request(fake):
    images.end_turn("t-many")
    try:
        msgs = [HumanMessage("browse")]
        for i in range(4):
            images.add_tool_pixels("t-many", f"s{i}", [{"type": "image", "base64": "eA==", "mime_type": "image/jpeg"}])
            msgs += [AIMessage(content="", tool_calls=[{"name": "browser_screenshot", "args": {}, "id": f"s{i}"}]),
                     ToolMessage(content="shot", tool_call_id=f"s{i}")]
        mw = [m for m in core._middleware() if getattr(m, "name", "") == "images"][0]
        runtime = type("R", (), {"execution_info": type("E", (), {"thread_id": "t-many"})()})()
        sent = mw._attach(type("Q", (), {"messages": msgs, "runtime": runtime})()).messages
        with_image = [m.tool_call_id for m in sent if isinstance(m, ToolMessage) and not isinstance(m.content, str)]
        assert with_image == ["s2", "s3"]
    finally:
        images.end_turn("t-many")


def test_a_conversations_browser_outlives_its_session_until_idle_or_cleared(fake):
    """The session cache rebuilds every minute for a user who has not consented; the page must
    not go with it. Idle ones are stopped from session lookups, and /clear stops its own."""
    closed = []
    with mock.patch.object(browser, "_browsers", {}), \
         mock.patch.object(browser, "_close_later", closed.append):
        a = browser.for_conversation("web:a", "t-1")
        assert browser.for_conversation("web:a", "t-1") is a           # a rebuild gets the same one
        a.client, a.last_used = _Client("r"), time.time() - browser._IDLE_SECONDS - 1
        b = browser.for_conversation("web:b", "t-2")
        b.client = _Client("r")
        browser.sweep()
        assert closed == [a]
        browser.close("web:b", "t-2")
        assert closed == [a, b] and browser.for_conversation("web:b", "t-2") is not b
