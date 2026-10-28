"""The tools a session is assembled with, and the components behind them: web search, memory."""

import asyncio
import os
import time
from unittest import mock

import pytest

import tools
from identity import grants, lark_3lo
from runtime import sessions
from tools import code_interpreter, mcp_client, memory, web_search

from support import _dummy_tool, _make_tool


def test_web_search_is_signed_by_the_agent_role_not_a_users_token():
    """Search reads no user data, and minting a per-user token for it would need the agent
    to hold Cognito admin rights — and create a Cognito user for every web actor."""
    with mock.patch.object(web_search, "_GATEWAY_URL",
                           "https://gw-1.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"):
        conn = web_search.connection_for("web:abc")
    assert "Authorization" not in conn.get("headers", {})
    assert conn["auth"]._signer._region_name == "us-east-1"
    assert conn["auth"]._signer._service_name == "bedrock-agentcore"
    with mock.patch.object(web_search, "_GATEWAY_URL", ""):
        assert web_search.available() is False      # no gateway: no tool, not an error


def test_unauthorized_session_passes_an_empty_token_and_still_gets_tools():
    """The token is passed as an empty header rather than skipping the connection —
    verified against the deployed Runtime, which returns the full tool list for an
    empty token. Skipping it would leave the model unaware of its Lark tools."""
    opened = []

    async def fake_open(stack, connection):
        opened.append(connection["headers"])
        return [_dummy_tool]

    with mock.patch.object(lark_3lo, "get_user_lark_token",
                           return_value=("auth_url", "https://consent")), \
         mock.patch.object(lark_3lo, "mcp_connection_for",
                           side_effect=lambda tok, url="": {"headers": {"tok": tok}}), \
         mock.patch.object(tools, "_aopen_tools", fake_open), \
         mock.patch.object(web_search, "available", return_value=False):
        s = sessions._build_session("lark:ou_x", "", "mem1")

    assert s["auth_url"] == "https://consent"      # consent is remembered, not raised
    assert s["consent"] == {"lark": "https://consent"}
    assert s["tool_idp"] == {_dummy_tool.name: "lark"}
    assert opened == [{"tok": ""}], "the server must still be connected, with no token"
    assert s["graph"] is not None, "the model must see the Lark tools regardless"


def test_every_configured_component_reaches_the_session_as_tools():
    """Search, memory, whoami and code execution are each one branch of tools.assemble; a
    branch that stops adding its tools leaves the agent without them and nothing fails."""
    async def fake_open(stack, connection):
        return [_make_tool("WebSearch___WebSearch")]
    sandbox = mock.Mock()
    with mock.patch.object(lark_3lo, "configured", return_value=False), \
         mock.patch.dict(grants._DOWNSTREAMS, {}, clear=True), \
         mock.patch.object(web_search, "available", return_value=True), \
         mock.patch.object(web_search, "connection_for", return_value={}), \
         mock.patch.object(tools, "_aopen_tools", fake_open), \
         mock.patch.object(memory, "tools_for", return_value=[_make_tool("remember")]), \
         mock.patch.object(code_interpreter, "available", return_value=True), \
         mock.patch.object(code_interpreter, "_Sandbox", return_value=sandbox), \
         mock.patch.object(code_interpreter, "tools_for", return_value=[_make_tool("run_code")]):
        s = sessions._build_session("web:s-1", "", "mem-all")
    assert s["tool_source"] == {"WebSearch___WebSearch": "web_search", "remember": "memory",
                                "whoami": "identity", "run_code": "code"}
    assert s["sandbox"] is sandbox      # kept, so closing the session stops it


# Two tools, not automatic extraction. The isolation that matters is the namespace: it is
# built from the bound actor, never from a model-supplied argument, so no prompt can make
# recall read another user's records.

def test_no_memory_resource_means_no_tools_rather_than_an_error():
    with mock.patch.object(memory, "_MEMORY_ID", ""):
        assert memory.tools_for("lark:ou_x") == []


def test_the_namespace_comes_from_the_bound_actor_not_from_arguments():
    calls = {}

    class _C:
        def batch_create_memory_records(self, **kw):
            calls["write"] = kw
            return {"successfulRecords": [{"memoryRecordId": "r1"}]}

        def retrieve_memory_records(self, **kw):
            calls["read"] = kw
            return {"memoryRecordSummaries": [{"content": {"text": "code name is X"}}]}

    with mock.patch.object(memory, "_MEMORY_ID", "mem-1"), \
         mock.patch.object(memory.boto3, "client", return_value=_C()):
        remember, recall = memory.tools_for("lark:ou_alice")
        assert remember.invoke({"fact": "likes tea"}) == "Saved."
        assert "code name is X" in recall.invoke({"query": "code name?"})

    assert calls["write"]["records"][0]["namespaces"] == ["/facts/lark:ou_alice"]
    assert calls["read"]["namespace"] == ["/facts/lark:ou_alice"][0]
    # No strategy id: measured that the service embeds and scores directly-written records
    # without one, so declaring a strategy would only add cost and constraints.
    assert "memoryStrategyId" not in calls["write"]["records"][0]
    assert set(remember.args) == {"fact"} and set(recall.args) == {"query"}


def test_a_rejected_record_is_reported_not_swallowed():

    class _C:
        def batch_create_memory_records(self, **kw):
            return {"successfulRecords": [], "failedRecords": [{"errorCode": "Throttled"}]}

    with mock.patch.object(memory, "_MEMORY_ID", "mem-1"), \
         mock.patch.object(memory.boto3, "client", return_value=_C()):
        remember, _ = memory.tools_for("lark:ou_x")
        assert "could not save" in remember.invoke({"fact": "x"})


def test_recall_says_so_when_there_is_nothing_stored():
    """An empty result must not read as a tool failure, or the model retries it."""

    class _C:
        def retrieve_memory_records(self, **kw):
            return {"memoryRecordSummaries": []}

    with mock.patch.object(memory, "_MEMORY_ID", "mem-1"), \
         mock.patch.object(memory.boto3, "client", return_value=_C()):
        _, recall = memory.tools_for("lark:ou_x")
        assert recall.invoke({"query": "anything?"}) == "Nothing on record about that."


@pytest.fixture(params=["modern", "legacy"])
def _mcp_server(request):
    """The fixture server, speaking both eras or the legacy handshake only."""
    import socket
    import subprocess
    import urllib.request
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    here = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.Popen(["node", os.path.join(here, "mcp_fixture_server.js")],
                            env={**os.environ, "PORT": str(port),
                                 "MODERN": "1" if request.param == "modern" else "0"})
    url = f"http://127.0.0.1:{port}/mcp"
    for _ in range(50):
        try:
            urllib.request.urlopen(url, timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    yield request.param, url
    proc.terminate()
    proc.wait(timeout=5)


def test_a_servers_tools_work_in_whichever_era_it_speaks(_mcp_server):
    """The client tries 2026-07-28 first and falls back to the handshake, so our servers (both
    eras) and older ones — the Gateway, GitHub's — are reached the same way. A failed call
    comes back as a failed tool output the model can read, not an exception ending the turn."""
    import json as _json
    import urllib.request
    from contextlib import AsyncExitStack
    from datetime import timedelta
    era, url = _mcp_server

    async def go():
        async with AsyncExitStack() as stack:
            got = {t.name: t for t in await mcp_client.open_tools(
                stack, {"transport": "streamable_http", "url": url, "timeout": timedelta(seconds=10)})}
            ok = await got["echo"].ainvoke({"name": "echo", "args": {"text": "hi"}, "id": "c1",
                                            "type": "tool_call"})
            failed = await got["fail"].ainvoke({"name": "fail", "args": {}, "id": "c2",
                                                "type": "tool_call"})
            return sorted(got), ok, failed
    names, ok, failed = asyncio.run(go())
    assert names == ["echo", "fail", "revoked"]
    assert [(b["type"], b["text"]) for b in ok.content] == [("text", "echo: hi")] and ok.status == "success"
    assert ok.artifact == {"structured_content": {"text": "hi"}}
    assert failed.status == "error" and "no user token (authorize first)" in str(failed.content)
    seen = _json.loads(urllib.request.urlopen(url, timeout=5).read())
    if era == "modern":
        assert [m for m, _ in seen][:1] == ["server/discover"] and "initialize" not in [m for m, _ in seen]
        assert {v for m, v in seen} == {"2026-07-28"}
    else:
        assert "initialize" in [m for m, _ in seen]
        assert {v for m, v in seen if m == "tools/call"} == {"2025-11-25"}


def test_a_server_that_turns_the_credentials_away_is_reported_not_raised(_mcp_server):
    """Under mcp 1.x a 401 left the call waiting forever; 2.x raises, which would end the turn.
    It comes back as failed output instead — for a vendor's server, worded by the caller so the
    consent wall takes over (tools.assemble passes the wall's marker)."""
    from contextlib import AsyncExitStack
    from datetime import timedelta
    _, url = _mcp_server
    conn = {"transport": "streamable_http", "url": url, "timeout": timedelta(seconds=10)}
    walled = []

    async def go(on_refused):
        async with AsyncExitStack() as stack:
            got = {t.name: t for t in await mcp_client.open_tools(stack, conn, on_refused)}
            return await got["revoked"].ainvoke({"name": "revoked", "args": {}, "id": "c3",
                                                 "type": "tool_call"})
    out = asyncio.run(go(lambda: walled.append(1) or "no user token (authorize first)"))
    assert out.status == "error" and "no user token (authorize first)" in str(out.content)
    assert walled == [1]
    out = asyncio.run(go(None))
    assert out.status == "error" and "refused this call's credentials" in str(out.content)


def test_a_runtime_hosted_server_is_reached_through_the_handshake():
    """The Runtime keeps a connection on one microVM only by the Mcp-Session-Id an `initialize`
    answer carries; under 2026-07-28 each call landed on a fresh one (measured). The Gateway
    keeps its own sessions with targets, so it is negotiated like any other server."""
    conn = lark_3lo.mcp_connection_for("tok", url="https://runtime/mcp")
    assert conn["mode"] == "legacy"
    assert web_search.connection_for("web:x").get("mode", "auto") == "auto"
