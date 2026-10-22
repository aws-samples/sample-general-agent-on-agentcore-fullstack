// Measurement MCP server: answers "is one runtime session served by one instance?"
//
// Built for the Runtime V2 question, where the usual discriminators stop working: a
// snapshot is copied to every restored instance, so hostname is always `localhost`,
// PID always `1`, and anything generated at start-up is identical everywhere. The only
// evidence that survives is state written *after* the restore — the same conclusion
// docs/agentcore-behavior.md reached for proving per-session isolation.
//
// So this reports four things at once and lets them disagree:
//   startupId  — generated at module load. Identical across restores on V2; that is the point.
//   calls      — incremented in memory per request. 1,2,3… on one instance; 1,1,1… if every
//                call lands on a fresh restore. This is the affinity answer.
//   marker     — whether a file this process wrote on its first request is still there.
//                A second opinion on `calls`, immune to a counter being reset.
//   kernelAge  — /proc/uptime, maintained by the kernel rather than by this process.
'use strict';

const http = require('http');
const fs = require('fs');
const crypto = require('crypto');

const PORT = parseInt(process.env.PORT || '8000', 10);
const STARTUP_ID = crypto.randomBytes(4).toString('hex');
const MARKER = `/tmp/probe-${STARTUP_ID}`;
let calls = 0;

function kernelAge() {
  try { return parseFloat(fs.readFileSync('/proc/uptime', 'utf8').split(' ')[0]); }
  catch { return null; }
}

function observe(sessionId) {
  calls += 1;
  const markerExisted = fs.existsSync(MARKER);
  if (!markerExisted) fs.writeFileSync(MARKER, String(Date.now()));
  return {
    startupId: STARTUP_ID,                       // snapshot-shared on V2
    requestId: crypto.randomBytes(4).toString('hex'),  // fresh entropy per request
    calls,                                       // the affinity signal
    markerExisted,                               // false only on this instance's first call
    kernelAgeSeconds: kernelAge(),
    pid: process.pid,
    hostname: require('os').hostname(),
    sessionId: sessionId ? sessionId.slice(0, 16) : null,
  };
}

const TOOL = {
  name: 'probe_instance',
  description: 'Report which runtime instance served this call.',
  inputSchema: { type: 'object', properties: {} },
};

function sse(res, obj) {
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
  res.end(`event: message\ndata: ${JSON.stringify(obj)}\n\n`);
}

http.createServer((req, res) => {  // nosemgrep: using-http-server — TLS terminated by the Runtime
  if (req.method === 'GET') { res.writeHead(200); res.end('ok'); return; }
  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', () => {
    const sid = req.headers['x-amzn-bedrock-agentcore-runtime-session-id'] || '';
    let mcp; try { mcp = JSON.parse(body); } catch { res.writeHead(400); res.end('bad json'); return; }
    if (mcp.method === 'initialize') {
      return sse(res, { jsonrpc: '2.0', id: mcp.id, result: {
        protocolVersion: '2025-11-25', capabilities: { tools: { listChanged: false } },
        serverInfo: { name: 'probe-mcp', version: '1.0.0' } } });
    }
    if (mcp.method === 'notifications/initialized') { res.writeHead(202); res.end(); return; }
    if (mcp.method === 'tools/list') {
      return sse(res, { jsonrpc: '2.0', id: mcp.id, result: { tools: [TOOL] } });
    }
    // Plain POST with no MCP method: the HTTP-protocol shape, which is how the agent
    // runtime itself is invoked. Session affinity is a property of that shape, not of a
    // per-call MCP connection, so the comparison has to be made here.
    if (!mcp.method) {
      const obs = observe(sid);
      console.log('probe ' + JSON.stringify(obs));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(obs));
      return;
    }
    if (mcp.method === 'tools/call') {
      const obs = observe(sid);
      console.log('probe ' + JSON.stringify(obs));
      return sse(res, { jsonrpc: '2.0', id: mcp.id, result: {
        content: [{ type: 'text', text: JSON.stringify(obs) }] } });
    }
    return sse(res, { jsonrpc: '2.0', id: mcp.id || null, error: { code: -32601, message: `method not found: ${mcp.method}` } });
  });
}).listen(PORT, '0.0.0.0', () => {  // nosec B104 # nosemgrep: avoid-bind-to-all-interfaces — the Runtime routes to this port
  console.log(`probe-mcp on :${PORT} startupId=${STARTUP_ID} node=${process.version}`);
});
