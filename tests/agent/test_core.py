"""The agent itself: prompt caching, the model's timeouts, the middleware it is built with."""

import os
from unittest import mock

import pytest
from langchain_core.messages import HumanMessage

import core
import middleware

from support import _CaptureClient, _ScriptedModel, _dummy_tool, _invoke_capturing


def test_the_agent_is_built_with_its_middleware():
    """build() is the one place the middleware is attached; dropped there, steering and
    summarisation stop together while every test that builds its own graph stays green."""
    from langgraph.checkpoint.memory import InMemorySaver
    with mock.patch.object(core, "_model", _ScriptedModel(turns=[{"text": ["ok"]}])):
        graph = core.build([_dummy_tool], InMemorySaver())
    assert {"steering.before_model", "SummarizationMiddleware.before_model"} <= set(
        graph.get_graph().nodes)


@pytest.mark.parametrize("ttl,counts,sent", [
    ("1h", (1, 1, True), ["1h"]),
    # Bedrock's default window: langchain-aws omits the field rather than send it.
    ("5m", (1, 1, True), [None]),
    # The escape hatch must really send none: a cachePoint costs a 1.25-2x write per miss.
    ("", (0, 0, False), []),
])
def test_cache_points_reach_the_request_on_system_tools_and_messages(ttl, counts, sent):
    got = _invoke_capturing(cache_ttl=ttl)
    assert (got["system"], got["tools"], got["messages"] >= 1) == counts
    assert got["ttls"] == sent


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
    assert core._caches_explicitly(model) is explicit


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
        m = importlib.reload(core)
        assert (m._CACHE_TTL, m.cache_mode(), m._model.cache_ttl) == (ttl, mode, ttl)
    importlib.reload(core)


def test_a_mistyped_cache_setting_stops_the_container():
    import importlib
    with mock.patch.dict(os.environ, {"PROMPT_CACHE_TTL": "1hr"}), pytest.raises(SystemExit):
        importlib.reload(core)
    importlib.reload(core)


def test_an_explicit_per_call_cache_control_is_not_overridden():
    from langchain_core.messages import SystemMessage
    client = _CaptureClient()
    m = core._CachingChatBedrockConverse(
        model="global.anthropic.claude-sonnet-5", region_name="us-west-2",
        client=client, cache_ttl="1h")
    m.invoke([SystemMessage("sys"), HumanMessage("a")], cache_control={"ttl": "5m"})
    assert [b["cachePoint"].get("ttl") for b in client.params["system"]
            if "cachePoint" in b] == [None]      # 5m → field omitted


def test_images_come_first_and_summarisation_can_be_switched_off():
    """First on purpose: summarisation works on the placeholders, never on the bytes, and
    its counter charges a flat 85 tokens per image. Without it, steering (a mid-turn message
    is not lost) and images (the model sees a screenshot at all) stay."""
    names = lambda: [type(m).__name__ for m in core._middleware()]
    assert names() == ["ImageMiddleware", "SteeringMiddleware", "ClockMiddleware",
                       "SummarizationMiddleware"]
    with mock.patch.object(middleware, "_SUMMARIZE_AT_TOKENS", 0):
        assert names() == ["ImageMiddleware", "SteeringMiddleware", "ClockMiddleware"]


def test_the_turn_model_outlasts_a_silent_tool_call():
    """A model can go quiet for over a minute while it writes one large tool argument;
    the read timeout must outlast that, while a dead connection still fails to connect fast."""
    client = core._model.client
    assert client.meta.config.read_timeout == core._MODEL_READ_TIMEOUT >= 300
    assert client.meta.config.connect_timeout == 60
