"""Web search, via the AgentCore Gateway's built-in connector.

Unlike the Lark tools, search needs no end-user identity: it asks Amazon's web
index, not the user's account. So it goes through the Gateway with
GATEWAY_IAM_ROLE outbound auth and needs none of the per-user 3LO machinery the
Lark tools carry (see docs/agentcore-behavior.md).

The connector is only offered in us-east-1, so this gateway typically lives in a
different region from everything else and is called cross-region.

Inbound auth is IAM: the agent signs with its own role. Search reads no user data, so the
Gateway has no use for a per-user token — and not minting one keeps the agent from holding
Cognito admin rights at all. Who searched is in the agent's own logs.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import timedelta

import boto3

import lark_3lo

log = logging.getLogger("agent.websearch")

_GATEWAY_URL = os.environ.get("WEBSEARCH_GATEWAY_URL", "")
_MCP_VERSION = "2025-11-25"  # the version the gateway was created with


def available() -> bool:
    return bool(_GATEWAY_URL)


def _gateway_region() -> str:
    """The region the URL names: the connector lives in us-east-1 whatever ours is, and a
    SigV4 signature for the wrong region is refused."""
    m = re.search(r"\.bedrock-agentcore\.([a-z0-9-]+)\.amazonaws\.com", _GATEWAY_URL)
    return m.group(1) if m else "us-east-1"


def connection_for(actor_id: str) -> dict:
    """A `StreamableHttpConnection` for the search gateway, signed with the agent's role.
    Credentials are resolved per call, as for the MCP servers (lark_3lo.mcp_connection_for)."""
    region = _gateway_region()
    creds = boto3.Session(region_name=region).get_credentials()
    return {
        "transport": "streamable_http",
        "url": _GATEWAY_URL,
        # The Gateway is stateless and returns no session id; without this header it
        # negotiates 2025-03-26 and rejects the request (-32022).
        "headers": {"MCP-Protocol-Version": _MCP_VERSION},
        "auth": lark_3lo._SigV4HTTPXAuth(creds, "bedrock-agentcore", region),
        "timeout": timedelta(seconds=60),
    }
