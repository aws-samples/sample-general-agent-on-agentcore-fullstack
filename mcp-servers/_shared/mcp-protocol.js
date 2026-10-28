// The MCP protocol every server here speaks, both eras on one endpoint. Copied into each
// server's image by scripts/build-mcp.sh; a server keeps only its own tools/list and tools/call.
//
// Legacy (2024-11-05 … 2025-11-25): an `initialize` handshake settles the version — the
// client's own when supported, else the newest legacy one. Modern (2026-07-28): no handshake;
// each request names its version in params._meta, `server/discover` says what is spoken, and
// results carry resultType. Mirrors the reference Python SDK 2.3 server, measured on the wire
// (see .dev/adr/0012).
'use strict';

const LEGACY = ['2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05'];  // newest first
const MODERN = ['2026-07-28'];
const VERSION_KEY = 'io.modelcontextprotocol/protocolVersion';
const SERVER_INFO_KEY = 'io.modelcontextprotocol/serverInfo';
const CAPABILITIES = { tools: { listChanged: false } };

// The version a modern request names, or '' for a request made after a legacy handshake.
function modernVersion(msg) {
  const meta = msg && msg.params && msg.params._meta;
  return meta && typeof meta[VERSION_KEY] === 'string' ? meta[VERSION_KEY] : '';
}

function decorate(msg, result, info) {
  const listing = msg.method === 'tools/list' || msg.method === 'server/discover';
  return {
    ...result,
    resultType: 'complete',
    ...(listing ? { cacheScope: 'private', ttlMs: 0 } : {}),
    _meta: { ...(result._meta || {}), [SERVER_INFO_KEY]: info },
  };
}

// What is protocol rather than tools — the handshake, discovery, a version this server does
// not speak — as { status, body? }, or null for the server's own handler. `info` is
// { name, version }.
function lifecycle(msg, info) {
  if (msg.method === 'initialize') {
    const asked = msg.params && msg.params.protocolVersion;
    return { status: 200, body: { jsonrpc: '2.0', id: msg.id, result: {
      protocolVersion: LEGACY.includes(asked) ? asked : LEGACY[0],
      capabilities: CAPABILITIES,
      serverInfo: info,
    } } };
  }
  if (msg.method === 'notifications/initialized') return { status: 202 };
  const version = modernVersion(msg);
  if (!version) return null;
  if (!MODERN.includes(version)) {
    return { status: 400, body: { jsonrpc: '2.0', id: msg.id ?? null, error: {
      code: -32022, message: 'Unsupported protocol version',
      data: { supported: MODERN, requested: version },
    } } };
  }
  if (msg.method === 'server/discover') {
    return { status: 200, body: { jsonrpc: '2.0', id: msg.id,
      result: decorate(msg, { supportedVersions: MODERN, capabilities: CAPABILITIES }, info) } };
  }
  return null;
}

// The server's own answer, shaped for the era the request was made in.
function reply(msg, payload, info) {
  if (!modernVersion(msg) || !payload.result) return payload;
  return { ...payload, result: decorate(msg, payload.result, info) };
}

module.exports = { lifecycle, reply, LEGACY, MODERN };
