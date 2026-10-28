"""The agent itself: which model it calls, its system prompt, the middleware it runs, and
`build()`, which puts them together with a session's tools into a LangGraph agent.

Start here. The rest of agent/ serves it: tools/ decides the tools (one module per AgentCore
component), identity/ whose credentials they use, runtime/ keeps one built agent per user and
conversation and runs one turn per conversation at a time, channels/ carries a turn in from Lark
or the web page and the answer back, and server.py / a2a_server.py are the Runtime contracts.
"""

from __future__ import annotations

import logging
import os

from botocore.config import Config
from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse

import middleware

log = logging.getLogger("agent.core")

_REGION = os.environ.get("AWS_REGION", "us-west-2")


_MODEL_ID = os.environ["BEDROCK_MODEL_ID"]  # injected at deploy, from MODEL_ID in .env


# Short on purpose. Operational facts a model cannot infer — the workspace persists, whether
# there is internet — live in each tool's description, next to what they constrain, instead of
# being paid for on every turn. Debugging heuristics do not belong here at all: they encode
# one deployment's incident history into a general-purpose agent.
_SYSTEM = os.environ.get(
    "AGENT_SYSTEM_PROMPT",
    "You are a general-purpose agent. Be concise. Use the tools when they help. "
    "Report what a tool actually returned, and never present work you did not "
    "complete as done.",
)


# Prompt caching on Bedrock is explicit or implicit by model, not by choice. On Converse only
# Claude and Nova take a cachePoint, and they cache nothing without one; every other model
# caches matching prefixes on its own or not at all, and may reject a request that carries a
# cachePoint (Kimi K3 does, measured). So the model decides whether to send one, and
# PROMPT_CACHE_TTL only sets how long: "5m" (Bedrock's default) | "1h" | "off".
# Source: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
# Placement, TTL choice and the measured break-even: docs/architecture.md.
_EXPLICIT_CACHE_FAMILIES = ("anthropic.claude", "amazon.nova")


def _caches_explicitly(model_id: str) -> bool:
    """Whether this model caches only behind a cachePoint (and accepts one)."""
    return any(f in model_id.lower() for f in _EXPLICIT_CACHE_FAMILIES)


_CACHE_SETTING = os.environ.get("PROMPT_CACHE_TTL", "").strip() or "5m"


if _CACHE_SETTING not in ("5m", "1h", "off"):
    raise SystemExit(f"PROMPT_CACHE_TTL must be 5m, 1h or off, not {_CACHE_SETTING!r}")


# The TTL actually put on requests: empty sends no cachePoint at all.
_CACHE_TTL = (_CACHE_SETTING if _caches_explicitly(_MODEL_ID) and _CACHE_SETTING != "off"
              else "")


class _CachingChatBedrockConverse(ChatBedrockConverse):
    """ChatBedrockConverse that asks for prompt caching on every request.

    langchain-aws does the placement AWS documents — after the system prompt, after the
    tool definitions, and a rolling pair at the end of the message list, up to Bedrock's
    limit of four. All we supply is the intent.

    Before tuning this, know that the 1,024-token minimum for Sonnet 5 is cumulative over
    tools + system + messages in that order. Measured against the deployed servers, our
    tool definitions are ~1,618 tokens with the approval server and ~400 without it, and
    the system prompt is ~32 — so on a minimal deployment the tools checkpoint earns
    nothing until the conversation itself grows past the minimum.
    """

    cache_ttl: str

    def _with_cache(self, kwargs: dict) -> dict:
        # An explicit per-call value wins: a caller asking for something specific should
        # not be silently overridden.
        if self.cache_ttl and "cache_control" not in kwargs:
            kwargs["cache_control"] = {"ttl": self.cache_ttl}
        return kwargs

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return super()._generate(messages, stop=stop, run_manager=run_manager,
                                 **self._with_cache(kwargs))

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        return super()._stream(messages, stop=stop, run_manager=run_manager,
                               **self._with_cache(kwargs))


# The class defines no _agenerate/_astream, so the async paths run these in a thread —
# overriding the two synchronous entry points covers streaming and non-streaming, sync
# and async alike.
# A model may send a tool call's arguments in one piece once they are all generated, so the
# stream is silent while it writes a large file: Kimi K3 sent 13 KB after 18.5 s of silence
# (measured), and botocore's 60 s default cut a longer one off. Connecting keeps the default.
_MODEL_READ_TIMEOUT = 300


_model = _CachingChatBedrockConverse(model=_MODEL_ID, region_name=_REGION,
                                     cache_ttl=_CACHE_TTL,
                                     config=Config(read_timeout=_MODEL_READ_TIMEOUT))


def _middleware() -> list:
    from langchain.agents.middleware import AgentMiddleware
    # Subclassed here rather than at import time so the class exists only when the
    # framework does, matching how summarisation is imported.
    steering = type("SteeringMiddleware", (AgentMiddleware,),
                    dict(middleware._SteeringMiddleware.__dict__))()
    # First in the list: the image bytes are attached to the outgoing request only, so
    # anything after this — summarisation especially — works on the placeholders.
    pictures = type("ImageMiddleware", (AgentMiddleware,),
                    dict(middleware._ImageMiddleware.__dict__))()
    clock = type("ClockMiddleware", (AgentMiddleware,),
                 dict(middleware._ClockMiddleware.__dict__))()
    if not middleware._SUMMARIZE_AT_TOKENS:
        return [pictures, steering, clock]
    return [pictures, steering, clock, middleware._summarization(_summary_model())]


def _summary_model():
    """The model that writes summaries, for the threshold and for /compact alike."""
    return _CachingChatBedrockConverse(
        model=_MODEL_ID, region_name=_REGION,
        # Caching deliberately off for this one: the summary call happens once per
        # threshold crossing and is never re-read, so a cachePoint would only buy a
        # 1.25-2x write premium.
        cache_ttl="",
        # The summary is generated inside the graph, so without these its tokens stream
        # into the chat like an answer: `nostream` keeps it out of LangGraph's messages
        # stream (the Lark card), `emit-messages` out of AG-UI (the web page).
        tags=["nostream"], metadata={"emit-messages": False},
        timeout=middleware._SUMMARY_TIMEOUT)


def model_id() -> str:
    """What this container actually calls Bedrock with — /status reads it from here so a
    stale router config cannot report a model nobody is using."""
    return _MODEL_ID


def cache_mode() -> str:
    """How this model's prompt cache works here: "explicit 5m" / "explicit 1h" (a cachePoint
    on every request), "implicit" (the model caches by itself) or "off"."""
    if _CACHE_TTL:
        return f"explicit {_CACHE_TTL}"
    return "off" if _caches_explicitly(_MODEL_ID) else "implicit"


def model_settings() -> dict:
    """What shapes a turn in this container, as it is actually configured. The prompt
    itself is not returned: it can be long, and its length says whether it was changed."""
    return {"model": _MODEL_ID, "region": _REGION, "cache": cache_mode(),
            "summarizeAt": middleware._SUMMARIZE_AT_TOKENS,
            "summarizeKeep": middleware._SUMMARIZE_KEEP,
            "systemPromptCustom": "AGENT_SYSTEM_PROMPT" in os.environ
                                  and bool(os.environ["AGENT_SYSTEM_PROMPT"].strip()),
            "systemPromptChars": len(_SYSTEM)}


def build(tools: list, checkpointer):
    """The agent: the model above, the system prompt, the middleware, and the tools this
    user's session was given (tools.assemble)."""
    # langchain.agents.create_agent, not langgraph.prebuilt.create_react_agent: the
    # latter is deprecated as of LangGraph 1.0 and slated for removal in 2.0 (AWS's
    # devguide example still imports it). Note there are no pre/post model hooks on
    # this API — middleware replaces them — which is another reason the conversational
    # record is written at turn end instead.
    return create_agent(model=_model, tools=tools, system_prompt=_SYSTEM,
                        checkpointer=checkpointer, middleware=_middleware())
