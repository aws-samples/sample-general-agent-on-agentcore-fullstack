// Minimal MCP server that executes the official lark-cli per tool call with the
// caller's per-request user_access_token — statelessly (env var, no AuthStore).
//
// Why this instead of official lark-mcp: lark-mcp validates a user token against
// its LOCAL AuthStore (populated only by `lark-mcp login`), so an externally
// vaulted token is rejected as "invalid or expired". lark-cli's env token
// (LARKSUITE_CLI_USER_ACCESS_TOKEN) bypasses that entirely — verified.
//
// Contract: AgentCore Runtime MCP server on 0.0.0.0:8000/mcp (streamable HTTP,
// SSE responses). Per-request user token arrives in the custom passthrough header
// the agent sets (no sidecar needed — we read it directly).

const http = require('http');
// In the image it sits beside this file (scripts/build-mcp.sh copies it in); in the repo, in _shared/.
const protocol = require(require('fs').existsSync(`${__dirname}/mcp-protocol.js`)
  ? './mcp-protocol' : '../_shared/mcp-protocol');
const { execFile } = require('child_process');

const SERVER_INFO = { name: 'lark-cli-mcp', version: '1.0.0' };
const PORT = parseInt(process.env.PORT || '8000', 10);
const APP_ID = process.env.APP_ID || '';
const APP_SECRET = process.env.APP_SECRET || '';
const BRAND = process.env.LARK_BRAND || 'lark';           // 'lark' (international) | 'feishu'
const TOKEN_HEADER = 'x-amzn-bedrock-agentcore-runtime-custom-lark-token';  // lowercased by node

// Every route to a Drive document asks the guard (guard.js) first: the document tools, and any
// `lark_api` call into a document family (routes.js), whose listings are filtered on the way
// back. Everything else in Lark passes untouched. See docs/architecture.md.
const { createGuard } = require('./guard');

const { TOKEN_RE, apiRoute, classify } = require('./routes');

const PROTECTED = (what) => `This ${what} is protected: the agent may not open it. Tell the user it is off limits.`;
const UNCHECKED = 'Protection rules could not be checked just now, so nothing was opened. Try again shortly.';
// One item's outcome from guard.decide → an error reply, or null to go ahead.
const refusal = (outcome, what) => (outcome === true ? null
  : { isError: true, text: outcome === false ? PROTECTED(what) : UNCHECKED });
// A listing keeps what was allowed, and says when something was left out and why.
function kept(entries, outcomes) {
  const keep = entries.filter((_, n) => outcomes[n] === true);
  const note = outcomes.includes(null) ? 'Some entries could not be checked against protection rules and are not shown.'
    : outcomes.includes(false) ? 'Some entries are protected and are not shown.' : '';
  return { keep, note };
}

// What a tool call can reach. `run(args, token)` resolves to { text, isError }.
const TOOLS = [
  {
    name: 'lark_whoami',
    description: "Return the calling Lark user's own profile (name, open_id, email). Proves the agent is acting as that user.",
    inputSchema: { type: 'object', properties: {} },
    run: async (a, token) => reply(await lark(['api', 'GET', '/open-apis/authen/v1/user_info'], token)),
  },
  {
    name: 'lark_list_my_docs',
    description: "List the calling user's own Lark cloud-drive files (scoped to what that user can see). Optional folder_token to list a subfolder. Entries protected from the agent are left out.",
    inputSchema: { type: 'object', properties: { folder_token: { type: 'string', description: 'optional folder token; omit for drive root' } } },
    run: async (a, token) => {
      if (a.folder_token && !TOKEN_RE.test(a.folder_token)) return { isError: true, text: 'folder_token is not a Lark token' };
      if (a.folder_token) {
        const no = refusal((await guard.decide(token, 'list', [{ kind: 'folder', token: a.folder_token }]))[0], 'folder');
        if (no) return no;
      }
      const params = { page_size: 200, ...(a.folder_token ? { folder_token: a.folder_token } : {}) };
      const r = await lark(['api', 'GET', '/open-apis/drive/v1/files', '--params', JSON.stringify(params)], token);
      if (!r.ok) return reply(r);
      const files = (r.data && r.data.files) || [];
      const { keep, note } = kept(files, await guard.decide(token, 'read', files.map(entryOf)));
      return reply({ ...r, data: { ...r.data, files: keep } }, note);
    },
  },
  {
    name: 'lark_read_document',
    description: "Read the text of a Lark document (docx) the calling user can see, by its token — the part after /docx/ in its link.",
    inputSchema: { type: 'object', properties: { document_token: { type: 'string', description: 'the docx token' } }, required: ['document_token'] },
    run: async (a, token) => {
      if (!TOKEN_RE.test(a.document_token || '')) return { isError: true, text: 'document_token is not a Lark token' };
      const no = refusal((await guard.decide(token, 'read', [{ kind: 'document', token: a.document_token }]))[0], 'document');
      if (no) return no;
      return reply(await lark(['api', 'GET', `/open-apis/docx/v1/documents/${a.document_token}/raw_content`], token));
    },
  },
  {
    name: 'lark_search_documents',
    description: "Search the documents the calling user can see by keyword. Protected documents are left out.",
    inputSchema: { type: 'object', properties: { query: { type: 'string' }, count: { type: 'integer', description: 'at most 20' } }, required: ['query'] },
    run: async (a, token) => {
      const body = { search_key: String(a.query || ''), count: Math.min(Math.max(parseInt(a.count, 10) || 10, 1), 20), offset: 0 };
      const r = await lark(['api', 'POST', '/open-apis/suite/docs-api/search/object', '--data', JSON.stringify(body)], token);
      if (!r.ok) return reply(r);
      const hits = (r.data && r.data.docs_entities) || [];
      const outcomes = await guard.decide(token, 'read', hits.map((h) => ({ kind: h.docs_type === 'folder' ? 'folder' : 'document', token: h.docs_token })));
      const { keep, note } = kept(hits, outcomes);
      return reply({ ...r, data: { ...r.data, docs_entities: keep } }, note);
    },
  },
  {
    name: 'lark_api',
    description: "Raw Lark OpenAPI passthrough as the calling user. method=GET/POST/... path=/open-apis/... ; optional params (query, JSON string) and data (body, JSON string).",
    inputSchema: {
      type: 'object',
      properties: {
        method: { type: 'string', description: 'HTTP method, e.g. GET or POST' },
        path: { type: 'string', description: 'API path, e.g. /open-apis/im/v1/chats' },
        params: { type: 'string', description: 'optional query params as a JSON string' },
        data: { type: 'string', description: 'optional request body as a JSON string' },
      },
      required: ['method', 'path'],
    },
    run: async (a, token) => {
      const route = apiRoute(a.method, a.path);
      if (route.error) return { isError: true, text: route.error };
      const doc = classify(route.method, route.path, a.params, a.data);
      if (doc.encoded) return { isError: true, text: 'Write the path without percent-encoding (for example a range as 0b12!A1:B2); an encoded document path cannot be checked, so it was not run.' };
      if (doc.document && !doc.known) {
        return { isError: true, text: `${route.method} ${route.path.split('?')[0]} reaches Drive documents but is not in the protection table, so it could not be checked and was not run.` };
      }
      if (doc.unreadable) return { isError: true, text: 'params and data must be JSON strings.' };
      if (doc.malformed) return { isError: true, text: `${doc.malformed} must be a Lark token (or a list of them), so the call could not be checked and was not run.` };
      if (doc.document && doc.items.length) {
        const outcomes = await guard.decide(token, doc.action, doc.items);
        const no = refusal(outcomes.find((o) => o !== true) ?? true, 'document or folder');
        if (no) return no;
      }
      const args = ['api', route.method, route.path];
      if (a.params) args.push('--params', String(a.params));
      if (a.data) args.push('--data', String(a.data));
      const r = await lark(args, token);
      const raw = doc.listing && r.ok && r.data && r.data[doc.listing.key];
      if (!raw || typeof raw !== 'object') return reply(r);
      // Listings and search come back filtered, as from the document tools. A listing is an
      // array, or a map keyed by token.
      const list = Array.isArray(raw) ? raw : Object.values(raw);
      const entries = list.map((e) => (e && e.shortcut_info ? entryOf(e) : {
        kind: doc.listing.type(e || {}) === 'folder' ? 'folder' : 'document', token: String(doc.listing.token(e || {}) || '') }));
      const { keep, note } = kept(list, await guard.decide(token, 'read', entries));
      const shown = Array.isArray(raw) ? keep
        : Object.fromEntries(Object.entries(raw).filter(([, e]) => keep.includes(e)));
      return reply({ ...r, data: { ...r.data, [doc.listing.key]: shown } }, note);
    },
  },
];

// A shortcut is judged by what it points at: its own token would let a protected document's
// name show through a shortcut placed in an unprotected folder.
const entryOf = (e) => {
  const target = e.type === 'shortcut' && e.shortcut_info;
  const type = target ? target.target_type : e.type;
  return { kind: type === 'folder' ? 'folder' : 'document', token: target ? target.target_token : e.token };
};

// lark-cli prints JSON with `ok`; success is that flag, not the presence of output. Hints that
// tell the reader to run `lark-cli auth login` are dropped: they would steer the model to a
// device-flow link instead of the consent the agent offers.
function reply(r, note = '') {
  if (r && r.error && typeof r.error === 'object') delete r.error.hint;
  const text = (typeof r === 'string' ? r : JSON.stringify(r, null, 2)) + (note ? `\n\n${note}` : '');
  return { isError: !(r && r.ok), text };
}

function lark(cliArgs, userToken) {
  return new Promise((resolve) => {
    const env = {
      PATH: process.env.PATH,
      HOME: process.env.HOME || '/tmp',
      LARKSUITE_CLI_USER_ACCESS_TOKEN: userToken,
      LARKSUITE_CLI_APP_ID: APP_ID,
      LARKSUITE_CLI_APP_SECRET: APP_SECRET,
      LARKSUITE_CLI_BRAND: BRAND,
      LARKSUITE_CLI_DEFAULT_AS: 'user',   // always act as the user, never the bot
    };
    execFile('lark-cli', cliArgs, { timeout: 30000, maxBuffer: 10 * 1024 * 1024, env },
      (err, stdout, stderr) => {
        const out = (stdout || '').trim();
        // Never stdout in the error: an unparsed listing would carry protected entries' names.
        try { resolve(JSON.parse(out)); } catch {
          resolve({ ok: false, error: { message: `lark-cli: ${(stderr || (err && err.message) || 'unreadable output').trim().slice(0, 500)}` } });
        }
      });
  });
}

const guard = (() => {
  const { VerifiedPermissionsClient, ListPoliciesCommand, BatchIsAuthorizedCommand } = require('@aws-sdk/client-verifiedpermissions');
  return createGuard({
    avp: new VerifiedPermissionsClient({}),
    commands: { ListPoliciesCommand, BatchIsAuthorizedCommand },
    lark,
    storeId: process.env.POLICY_STORE_ID,
    adminTemplateId: process.env.PROTECT_ADMIN_TEMPLATE_ID,
    userTemplateId: process.env.PROTECT_USER_TEMPLATE_ID,
    appId: APP_ID,
  });
})();

function sse(res, obj, status = 200) {
  res.writeHead(status, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
  res.end(`event: message\ndata: ${JSON.stringify(obj)}\n\n`);
}

const server = http.createServer((req, res) => {  // nosemgrep: using-http-server — TLS terminated by the Runtime
  if (req.method === 'GET') { res.writeHead(200); res.end('ok'); return; }
  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', async () => {
    const userToken = req.headers[TOKEN_HEADER] || '';
    // One line per inbound request: the MCP method and whether a user token rode along —
    // names and presence only, never a token value. A bare http server keeps no access log,
    // so without this "did the caller even reach us, and with what" is unanswerable; it has
    // been the first question in every routing/auth incident here. Logged before parsing so
    // an unparseable body is visible too — that is the case worth seeing, not the one to
    // drop silently.
    let mcp;
    try {
      mcp = JSON.parse(body);
    } catch {
      console.log(`inbound ${req.method} ${req.url} unparseable body (${body.length}B) `
        + `content-type=${req.headers['content-type'] || '(none)'} token=${userToken ? 'yes' : 'no'}`);
      res.writeHead(400); res.end('bad json'); return;
    }
    // Scheme only, never the credential: AWS4-HMAC-SHA256 means the transport signed this,
    // Bearer means a per-user token was injected — the distinction this log exists for.
    const authScheme = String(req.headers['authorization'] || '').split(' ')[0] || '(none)';
    console.log(`inbound ${req.method} ${req.url} mcp=${mcp.method || '(none)'} `
      + `token=${userToken ? 'yes' : 'no'} auth=${authScheme}`);

    const early = protocol.lifecycle(mcp, SERVER_INFO);
    if (early) {
      if (!early.body) { res.writeHead(early.status); res.end(); return; }
      return sse(res, early.body, early.status);
    }
    const send = (payload) => sse(res, protocol.reply(mcp, payload, SERVER_INFO));
    if (mcp.method === 'tools/list') {
      const tools = TOOLS.map((t) => ({ name: t.name, description: t.description, inputSchema: t.inputSchema }));
      return send({ jsonrpc: '2.0', id: mcp.id, result: { tools } });
    }
    if (mcp.method === 'tools/call') {
      const name = mcp.params && mcp.params.name;
      const args = (mcp.params && mcp.params.arguments) || {};
      const tool = TOOLS.find((t) => t.name === name);
      if (!tool) return send({ jsonrpc: '2.0', id: mcp.id, result: { content: [{ type: 'text', text: `unknown tool: ${name}` }], isError: true } });
      if (!userToken) return send({ jsonrpc: '2.0', id: mcp.id, result: { content: [{ type: 'text', text: 'no user token (authorize first)' }], isError: true } });
      const out = await tool.run(args, userToken);
      return send({ jsonrpc: '2.0', id: mcp.id, result: { content: [{ type: 'text', text: out.text }], isError: out.isError } });
    }
    return send({ jsonrpc: '2.0', id: mcp.id || null, error: { code: -32601, message: `method not found: ${mcp.method}` } });
  });
});

// Report the engine versions actually running, not the ones the build asked for: the
// image tag records the build argument, which is a different claim.
server.listen(PORT, '0.0.0.0', () => {
  console.log(`lark-cli-mcp on :${PORT} (${TOOLS.length} tools) node=${process.version} protection=${guard.configured ? 'on' : 'NOT CONFIGURED (document tools refuse)'}`);
  execFile('lark-cli', ['--version'], { timeout: 10000 }, (err, stdout) => {
    console.log(`lark-cli=${err ? `unavailable (${err.message})` : stdout.trim()}`);
  });
});
