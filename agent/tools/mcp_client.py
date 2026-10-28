"""MCP servers' tools as LangChain tools, on the MCP Python SDK 2.x.

langchain-mcp-adapters pins mcp<2, so this is the part of it the agent used: open a
connection, list the server's tools, call one, and convert the result the same way —
content blocks as the tool output, structuredContent as the artifact, an isError result as
a failed tool output the model reads rather than an exception that ends the turn. A
connection is the dict the agent already builds (`transport`, `url`, `headers`, `auth`,
`timeout`), so nothing that names a server changed. See .dev/adr/0012.

The protocol era is negotiated per server (`mode="auto"`): one that speaks 2026-07-28 is used
in it, an older one through the legacy handshake. A connection may name its mode instead — an
AgentCore-hosted server is reached through the handshake, which is what keeps it on one microVM
(identity/lark_3lo.mcp_connection_for).
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Callable

import httpx2
from langchain_core.messages.content import (create_file_block, create_image_block,
                                             create_text_block)
from langchain_core.tools import StructuredTool, ToolException
from mcp.client.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
import mcp_types as types

log = logging.getLogger("agent.mcp")

# A tool result streams back as SSE; how long to wait between its bytes. Under mcp 1.x a
# non-2xx answer left the call waiting forever, so every request now has a ceiling.
_READ_TIMEOUT = 300
_MAX_PAGES = 100


async def open_tools(stack: AsyncExitStack, connection: dict,
                     on_refused: Callable[[], str] | None = None) -> list[StructuredTool]:
    """Connect to one server for the session's lifetime and return its tools. The stack owns
    the teardown, so one close() releases the HTTP client and the session together.

    `on_refused` is called when the server turns the connection's credentials away (HTTP 401
    or 403 on a call) and returns what the model is told — for a vendor's server, the consent
    wall's own wording, so the turn ends asking the user to authorise again."""
    if connection.get("transport") not in ("streamable_http", "streamable-http", "http"):
        raise ValueError(f"unsupported MCP transport: {connection.get('transport')!r}")
    # The SDK reports any non-2xx answer as one generic error, so the status is read here.
    refused = {"seen": False}

    async def watch(response) -> None:
        if response.status_code in (401, 403):
            refused["seen"] = True
    timeout = connection.get("timeout")
    connect = timeout.total_seconds() if isinstance(timeout, timedelta) else float(timeout or 30)
    http = await stack.enter_async_context(httpx2.AsyncClient(
        headers=connection.get("headers") or {}, auth=connection.get("auth"),
        timeout=httpx2.Timeout(connect, read=_READ_TIMEOUT), follow_redirects=True,
        event_hooks={"response": [watch]}))
    client = await stack.enter_async_context(Client(
        streamable_http_client(connection["url"], http_client=http),
        mode=connection.get("mode", "auto"), read_timeout_seconds=_READ_TIMEOUT))
    return [_as_tool(client, t, refused, on_refused) for t in await _list_all(client)]


async def _list_all(client: Client) -> list[types.Tool]:
    tools, cursor = [], None
    for _ in range(_MAX_PAGES):
        page = await client.list_tools(cursor=cursor)
        tools.extend(page.tools or [])
        if not page.next_cursor:
            return tools
        cursor = page.next_cursor
    raise RuntimeError(f"tools/list did not end within {_MAX_PAGES} pages")


class _ToolError(ToolException):
    """An isError result. Its content goes back to the model as a failed tool output, so the
    agent can recover; a transport failure is not one of these and still raises."""

    def __init__(self, blocks: list[dict]) -> None:
        texts = [b["text"] for b in blocks if b.get("type") == "text"]
        super().__init__("\n".join(texts) or "The tool reported an error with no text.")
        self.blocks = blocks


def _failed_output(error: ToolException) -> list[dict] | str:
    if isinstance(error, _ToolError):
        return error.blocks or [create_text_block(text=str(error))]
    raise error


def _block(content) -> dict:
    """One MCP content item as a LangChain content block."""
    if isinstance(content, types.TextContent):
        return create_text_block(text=content.text)
    if isinstance(content, types.ImageContent):
        return create_image_block(base64=content.data, mime_type=content.mime_type)
    if isinstance(content, types.ResourceLink):
        mime = content.mime_type or None
        if mime and mime.startswith("image/"):
            return create_image_block(url=str(content.uri), mime_type=mime)
        return create_file_block(url=str(content.uri), mime_type=mime)
    if isinstance(content, types.EmbeddedResource):
        resource = content.resource
        if isinstance(resource, types.TextResourceContents):
            return create_text_block(text=resource.text)
        mime = resource.mime_type or None
        if mime and mime.startswith("image/"):
            return create_image_block(base64=resource.blob, mime_type=mime)
        return create_file_block(base64=resource.blob, mime_type=mime)
    # Audio, or a kind this was written before: named rather than dropped, so the model knows
    # something came back that it cannot see.
    return create_text_block(text=f"[{getattr(content, 'type', type(content).__name__)} content omitted]")


def _as_tool(client: Client, tool: types.Tool, refused: dict,
             on_refused: Callable[[], str] | None) -> StructuredTool:
    async def call(**arguments) -> tuple[list[dict], dict | None]:
        try:
            result = await client.call_tool(tool.name, arguments)
        except MCPError as e:
            # Failed output, not an exception: raised, it ends the turn, and a revoked grant
            # must end in a consent prompt instead.
            if refused["seen"]:
                text = on_refused() if on_refused else "The server refused this call's credentials."
            else:
                text = f"The {tool.name} call failed on the server: {e}"
            raise _ToolError([create_text_block(text=text)]) from e
        blocks = [_block(c) for c in result.content or []]
        if result.is_error:
            raise _ToolError(blocks)
        artifact = ({"structured_content": result.structured_content}
                    if result.structured_content is not None else None)
        return blocks, artifact

    return StructuredTool(
        name=tool.name, description=tool.description or "", args_schema=tool.input_schema,
        coroutine=call, response_format="content_and_artifact",
        handle_tool_error=_failed_output)
