// The protocol every MCP server here shares: both eras, answered as the reference SDK does.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const protocol = createRequire(import.meta.url)('../../mcp-servers/_shared/mcp-protocol.js');
const INFO = { name: 'test-mcp', version: '1.0.0' };
const modern = (method, version = '2026-07-28', extra = {}) => ({
  jsonrpc: '2.0', id: 7, method,
  params: { ...extra, _meta: { 'io.modelcontextprotocol/protocolVersion': version } },
});

test('a legacy handshake gets the client\'s own version when supported, else the newest', () => {
  const ask = (v) => protocol.lifecycle({ jsonrpc: '2.0', id: 1, method: 'initialize',
    params: { protocolVersion: v } }, INFO).body.result.protocolVersion;
  assert.equal(ask('2025-06-18'), '2025-06-18');
  assert.equal(ask('2025-11-25'), '2025-11-25');
  assert.equal(ask('2099-01-01'), '2025-11-25');
  assert.deepEqual(protocol.lifecycle({ method: 'notifications/initialized' }, INFO), { status: 202 });
});

test('a request after a legacy handshake is left to the server and answered as it was', () => {
  const msg = { jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} };
  assert.equal(protocol.lifecycle(msg, INFO), null);
  const payload = { jsonrpc: '2.0', id: 2, result: { tools: [] } };
  assert.deepEqual(protocol.reply(msg, payload, INFO), payload);
});

test('discovery says what is spoken, in the modern result shape', () => {
  const { status, body } = protocol.lifecycle(modern('server/discover'), INFO);
  assert.equal(status, 200);
  assert.deepEqual(body.result.supportedVersions, ['2026-07-28']);
  assert.deepEqual(body.result.capabilities, { tools: { listChanged: false } });
  assert.equal(body.result.resultType, 'complete');
  assert.deepEqual(body.result._meta['io.modelcontextprotocol/serverInfo'], INFO);
});

test('a modern version this server does not speak is a 400 naming what it does', () => {
  for (const method of ['server/discover', 'tools/list', 'tools/call']) {
    const { status, body } = protocol.lifecycle(modern(method, '2027-01-01'), INFO);
    assert.equal(status, 400, method);
    assert.deepEqual(body.error, { code: -32022, message: 'Unsupported protocol version',
      data: { supported: ['2026-07-28'], requested: '2027-01-01' } });
  }
});

test('a modern answer carries resultType and serverInfo, and a listing says it may be cached', () => {
  const list = protocol.reply(modern('tools/list'), { jsonrpc: '2.0', id: 7, result: { tools: [] } }, INFO);
  assert.equal(list.result.resultType, 'complete');
  assert.equal(list.result.cacheScope, 'private');
  assert.equal(list.result.ttlMs, 0);
  const call = protocol.reply(modern('tools/call', '2026-07-28', { name: 'x' }),
    { jsonrpc: '2.0', id: 7, result: { content: [], isError: false } }, INFO);
  assert.equal(call.result.resultType, 'complete');
  assert.equal('cacheScope' in call.result, false);
  assert.deepEqual(call.result._meta['io.modelcontextprotocol/serverInfo'], INFO);
  const failed = { jsonrpc: '2.0', id: 7, error: { code: -32601, message: 'method not found' } };
  assert.deepEqual(protocol.reply(modern('nope'), failed, INFO), failed);
});
