"""Puts agent/ on the path and supplies the env a deploy injects: the modules read it at import
and have no fallback."""

import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "agent")))
os.environ.setdefault("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-5")
os.environ.setdefault("LARK_API_DOMAIN", "https://open.larksuite.com")
os.environ.setdefault("LARK_OAUTH_PROVIDER", "test-3lo")
os.environ.setdefault("AGENT_WORKLOAD_NAME", "test-wl")
os.environ.setdefault("METRICS_NAMESPACE", "test/agent")
