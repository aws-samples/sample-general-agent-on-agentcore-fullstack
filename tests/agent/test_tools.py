"""The tools a session is assembled with, and the components behind them: web search, memory."""

from unittest import mock

import tools
from identity import grants, lark_3lo
from runtime import sessions
from tools import code_interpreter, memory, web_search

from support import _dummy_tool, _make_tool


def test_web_search_is_signed_by_the_agent_role_not_a_users_token():
    """Search reads no user data, and minting a per-user token for it would need the agent
    to hold Cognito admin rights — and create a Cognito user for every web actor. Without the
    protocol-version header the Gateway negotiates 2025-03-26 and rejects with -32022."""
    with mock.patch.object(web_search, "_GATEWAY_URL",
                           "https://gw-1.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"):
        conn = web_search.connection_for("web:abc")
    assert "Authorization" not in conn["headers"]
    assert conn["headers"]["MCP-Protocol-Version"] == "2025-11-25"
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
