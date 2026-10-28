"""What the agent suites share: a scripted model, sessions over real graphs, a recording
Lark card."""

import asyncio
import json
import os
from unittest import mock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool

import core
import middleware
import server
from channels import images, lark, lark_card
from runtime import sessions, turns

AGENT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "agent"))


def _repair(messages):
    """Drive the real repair against a graph whose state is `messages`, returning the
    count and whatever it appended."""
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
    n = asyncio.run(sessions._arepair_interrupted_turn(_Graph(), config))
    return n, appended


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
    script = [{"text": list(deltas), "tool_calls": list(tool_calls or [])}]
    if tool_calls:
        script.append({"text": [follow_up] if follow_up else []})
    tools = [_dummy_tool] + [_make_tool(n) for n in (tool_calls or [])]
    graph = core.create_agent(model=_ScriptedModel(turns=script), tools=tools)
    s = {"graph": graph, "tool_texts": [],
         "config": {"configurable": {"thread_id": f"t{next(_thread_seq)}"}}}
    s.update(extra)
    return s


def _patch_notify(card, notify_sent):
    return mock.patch.object(
        lark, "lark_card",
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


def _load_card(monkeypatch_calls):
    """Import StreamingCard with its HTTP layer replaced. lark_card imports boto3
    only, so unlike the agent modules it loads on the test host."""
    import importlib
    importlib.reload(lark_card)
    lark_card._tenant_token = lambda: "tok"

    def fake_call(method, url, body, bearer=""):
        content = body.get("content")
        if content is None and "elements" in body:       # a block's first write
            content = json.loads(body["elements"])[0].get("content")
        # Only the answer's text is under test here; status writes are counted elsewhere.
        if "/elements/p" in url or "/elements/tail" in url or url.endswith("/settings"):
            content = None
        monkeypatch_calls.append((method, content, body.get("sequence")))
        return {"code": 0}
    lark_card._call = fake_call
    return lark_card


def _recording_card():
    """A StreamingCard whose writes are recorded as (method, path, body)."""
    import importlib
    importlib.reload(lark_card)
    lark_card._tenant_token = lambda: "tok"
    writes = []

    def fake_call(method, url, body, bearer=""):
        writes.append((method, url.split("/cards/c1", 1)[-1], body))
        return {"code": 0}
    lark_card._call = fake_call
    card = lark_card.StreamingCard("oc_1")
    card.card_id, card.ok, card.sent = "c1", True, True
    return lark_card, card, writes


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


def _compactable(n_turns: int, summary: str):
    """A real agent graph with `n_turns` finished turns, and the summary model replaced."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langgraph.checkpoint.memory import InMemorySaver
    graph = core.create_agent(model=_ScriptedModel(turns=[{"text": ["ans"]}]),
                              tools=[], checkpointer=InMemorySaver(),
                              middleware=core._middleware()[:2])
    config = {"configurable": {"thread_id": f"c{next(_thread_seq)}"}}
    for i in range(n_turns):
        turns._run(graph.ainvoke({"messages": [HumanMessage(f"q{i}")]}, config=config))
    session = {"graph": graph, "config": config}
    return (session, mock.patch.object(sessions, "_get_session", return_value=session),
            mock.patch.object(core, "_summary_model",
                              return_value=FakeListChatModel(responses=[summary])))


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
    m = core._CachingChatBedrockConverse(
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


def _runtime(thread_id):
    """The runtime's real shape on LangGraph 1.2: the thread id is on execution_info."""
    return type("R", (), {"execution_info": type("E", (), {"thread_id": thread_id})()})()


def _image_mw():
    return [m for m in core._middleware() if getattr(m, "name", "") == "images"][0]


_BLOCK = {"type": "image", "base64": "Zm9v", "mime_type": "image/jpeg"}


def _register(thread_id, handles):
    """Register attachments with a turn, with the Lark fetch replaced."""
    with mock.patch.object(images, "blocks_for", return_value=[_BLOCK]):
        return images.add_to_turn(thread_id, handles)


def _fresh_claims():
    """Tests share the module's claim state; start each from nothing."""
    turns._in_flight = 0
    turns._in_flight_threads.clear()
    turns._pending_steer.clear()


class _InlineThread:
    """threading.Thread that runs its target on start(), so a turn finishes in the test."""
    def __init__(self, target=None, name=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


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
    with mock.patch.object(middleware, "_SUMMARIZE_AT_TOKENS", 0):
        base = core._middleware()                 # images, steering, clock
    with mock.patch.object(middleware, "_SUMMARIZE_AT_TOKENS", 1000), \
         mock.patch.object(middleware, "_SUMMARIZE_KEEP", 2):
        summariser = middleware._summarization(summary_model)
    graph = core.create_agent(
        model=_ScriptedModel(turns=[{"text": [answer]}]), tools=[_dummy_tool],
        middleware=base + [summariser], checkpointer=InMemorySaver())
    thread = f"t{next(_thread_seq)}"
    config = {"configurable": {"thread_id": thread}}
    graph.update_state(config, {"messages": _agentic_history()})
    return {"graph": graph, "tool_texts": [], "config": config}, thread


def _summary_update(text):
    from langchain_core.messages import RemoveMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), HumanMessage(
        content=f"Here is a summary of the conversation to date:\n\n{text}",
        additional_kwargs={"lc_source": "summarization"})]}


class _CapturedResp:
    def __init__(self, **kw): self.out = b""
    async def prepare(self, _r): pass
    async def write(self, b): self.out += b
    async def write_eof(self): pass

    def frames(self):
        return [f[len("data: "):] for f in self.out.decode().split("\n\n") if f]


# Any allowlisted user can take their own token from the web page and call the Runtime
# directly, so the actor comes from a verified bearer and the payload may only agree with it.



def _bind(headers, payload):
    from multidict import CIMultiDict
    request = mock.Mock(headers=CIMultiDict(headers))
    return server._bind_caller(request, payload)
