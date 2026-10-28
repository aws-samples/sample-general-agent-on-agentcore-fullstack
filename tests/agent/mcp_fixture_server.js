// A two-tool MCP server for tests/agent: MODERN=1 answers through the shared protocol module
// (both eras), otherwise it speaks the legacy handshake only, as a pre-2026 server does.
// GET /log returns the methods it saw, with the protocol-version header each carried.
const http = require('http');
const protocol = require('../../mcp-servers/_shared/mcp-protocol.js');
const INFO = { name: 'fixture', version: '0' };
const seen = [];
const TOOLS = [
  { name: 'echo', description: 'Echo back.', inputSchema: { type: 'object', properties: { text: { type: 'string' } } } },
  { name: 'fail', description: 'Always fails.', inputSchema: { type: 'object', properties: {} } },
  { name: 'revoked', description: 'Answers 401, as GitHub does for a revoked token.', inputSchema: { type: 'object', properties: {} } },
];
const sse = (res, obj, status = 200) => { res.writeHead(status, { 'Content-Type': 'text/event-stream' }); res.end(`event: message\ndata: ${JSON.stringify(obj)}\n\n`); };
http.createServer((req, res) => {
  if (req.method === 'GET') { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(seen)); return; }
  let body = ''; req.on('data', (c) => (body += c)); req.on('end', () => {
    const msg = JSON.parse(body);
    seen.push([msg.method, req.headers['mcp-protocol-version'] || '']);
    if (process.env.MODERN === '1') {
      const early = protocol.lifecycle(msg, INFO);
      if (early) { if (!early.body) { res.writeHead(early.status); res.end(); return; } return sse(res, early.body, early.status); }
    } else {
      if (msg.method === 'initialize') return sse(res, { jsonrpc: '2.0', id: msg.id, result: { protocolVersion: '2025-11-25', capabilities: { tools: {} }, serverInfo: INFO } });
      if (msg.method === 'notifications/initialized') { res.writeHead(202); res.end(); return; }
    }
    const send = (payload) => sse(res, protocol.reply(msg, payload, INFO));
    if (msg.method === 'tools/list') return send({ jsonrpc: '2.0', id: msg.id, result: { tools: TOOLS } });
    if (msg.method === 'tools/call') {
      const { name, arguments: args = {} } = msg.params;
      if (name === 'revoked') { res.writeHead(401, { 'Content-Type': 'application/json' }); res.end('{"message":"Bad credentials"}'); return; }
      if (name === 'fail') return send({ jsonrpc: '2.0', id: msg.id, result: { content: [{ type: 'text', text: 'no user token (authorize first)' }], isError: true } });
      return send({ jsonrpc: '2.0', id: msg.id, result: { content: [{ type: 'text', text: `echo: ${args.text}` }], structuredContent: { text: args.text }, isError: false } });
    }
    return send({ jsonrpc: '2.0', id: msg.id ?? null, error: { code: -32601, message: `method not found: ${msg.method}` } });
  });
}).listen(parseInt(process.env.PORT, 10), '127.0.0.1');
