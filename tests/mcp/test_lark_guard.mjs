// The lark-cli protection guard and the lark_api allowlist, against a fake AVP that applies the
// Cedar semantics the guard relies on: forbid wins, `in` is transitive over the entities the
// guard supplies, and a policy error is reported per decision.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { createGuard } = require('../../mcp-servers/lark-cli/guard.js');
const { apiRoute, classify } = require('../../mcp-servers/lark-cli/routes.js');

class ListPoliciesCommand { constructor(input) { this.input = input; } }
class BatchIsAuthorizedCommand { constructor(input) { this.input = input; } }
const commands = { ListPoliciesCommand, BatchIsAuthorizedCommand };

// rules: [{ template: 'admin' | 'user', principal?: 'app/ou', resource: {entityType, entityId} }]
function fakeAvp(rules, { errorsOn = null, throws = false } = {}) {
  const calls = { list: 0, batch: 0 };
  return {
    calls,
    async send(cmd) {
      if (throws) throw new Error('AVP unreachable');
      if (cmd instanceof ListPoliciesCommand) {
        calls.list++;
        const f = cmd.input.filter;
        const policies = rules
          .filter((r) => r.template === f.policyTemplateId
            && (!f.principal || f.principal.identifier.entityId === r.principal))
          .map((r) => ({ resource: r.resource, ...(r.principal ? { principal: { entityType: 'Lark::User', entityId: r.principal } } : {}) }));
        return { policies };
      }
      calls.batch++;
      const parents = new Map(cmd.input.entities.entityList.map((e) =>
        [`${e.identifier.entityType}/${e.identifier.entityId}`, e.parents.map((p) => `${p.entityType}/${p.entityId}`)]));
      const ancestors = (key, seen = new Set()) => {
        for (const p of parents.get(key) || []) if (!seen.has(p)) { seen.add(p); ancestors(p, seen); }
        return seen;
      };
      return {
        results: cmd.input.requests.map((q) => {
          const key = `${q.resource.entityType}/${q.resource.entityId}`;
          if (errorsOn === q.resource.entityId) return { request: q, decision: 'ALLOW', errors: [{ errorDescription: 'attribute missing' }] };
          const within = ancestors(key).add(key);
          const forbidden = rules.some((r) => (r.template === 'admin' || r.principal === q.principal.entityId)
            && within.has(`${r.resource.entityType}/${r.resource.entityId}`));
          return { request: q, decision: forbidden ? 'DENY' : 'ALLOW', errors: [] };
        }),
      };
    },
  };
}

// A drive: root → fld_A → fld_B → doc_deep; fld_A → doc_A; root → doc_top. Owners by token.
const DRIVE = { fld_A: [['fld_B', 'folder'], ['doc_A', 'docx']], fld_B: [['doc_deep', 'docx']] };
function fakeLark({ owners = { tokAlice: 'ou_alice', tokBob: 'ou_bob' }, drive = DRIVE, failList = 0 } = {}) {
  const calls = [];
  const fn = async (args, token) => {
    calls.push(args.join(' '));
    if (args[2] === '/open-apis/authen/v1/user_info') {
      return owners[token] ? { ok: true, data: { open_id: owners[token] } } : { ok: false, error: { code: 99991663 } };
    }
    if (args[2] === '/open-apis/drive/v1/files') {
      if (failList) return { ok: false, error: { code: failList, message: 'listing failed' } };
      const folder = JSON.parse(args[4]).folder_token;
      return { ok: true, data: { files: (drive[folder] || []).map(([token, type]) => ({ token, type })), has_more: false } };
    }
    throw new Error(`unexpected lark call ${args.join(' ')}`);
  };
  fn.calls = calls;
  return fn;
}

const IDS = { storeId: 'ps-1', adminTemplateId: 'admin', userTemplateId: 'user', appId: 'cli_app' };
const guardWith = (avp, lark, extra = {}) => createGuard({ avp, commands, lark, ...IDS, log: () => {}, ...extra });
const d = (token) => ({ kind: 'document', token });
const f = (token) => ({ kind: 'folder', token });

test('with no rules everything the user can see is allowed', async () => {
  const g = guardWith(fakeAvp([]), fakeLark());
  assert.deepEqual(await g.decide('tokAlice', 'read', [d('doc_top'), f('fld_A')]), [true, true]);
});

test("a user's document rule stops that user's agent only", async () => {
  const rules = [{ template: 'user', principal: 'cli_app/ou_alice', resource: { entityType: 'Lark::Document', entityId: 'doc_top' } }];
  const g = guardWith(fakeAvp(rules), fakeLark());
  assert.deepEqual(await g.decide('tokAlice', 'read', [d('doc_top'), d('doc_A')]), [false, true]);
  assert.deepEqual(await g.decide('tokBob', 'read', [d('doc_top')]), [true]);
});

test("an admin's folder rule covers everything beneath it, for every user", async () => {
  const rules = [{ template: 'admin', resource: { entityType: 'Lark::Folder', entityId: 'fld_A' } }];
  const g = guardWith(fakeAvp(rules), fakeLark());
  for (const tok of ['tokAlice', 'tokBob']) {
    assert.deepEqual(await g.decide(tok, 'read', [d('doc_deep'), d('doc_A'), f('fld_B'), d('doc_top')]),
      [false, false, false, true]);
    assert.deepEqual(await g.decide(tok, 'list', [f('fld_A')]), [false]);
  }
});

test('anything undecided is a refusal, reported as unchecked rather than protected', async () => {
  const rules = [{ template: 'admin', resource: { entityType: 'Lark::Folder', entityId: 'fld_A' } }];
  const cases = {
    'a protected folder listing hits a rate limit': guardWith(fakeAvp(rules), fakeLark({ failList: 99991400 })),
    'AVP is unreachable': guardWith(fakeAvp(rules, { throws: true }), fakeLark()),
    "the token's owner cannot be read": guardWith(fakeAvp(rules), fakeLark({ owners: {} })),
    'protection is not configured': guardWith(fakeAvp([]), fakeLark(), { storeId: '' }),
  };
  for (const [why, g] of Object.entries(cases)) {
    assert.deepEqual(await g.decide('tokAlice', 'read', [d('doc_top')]), [null], why);
  }
});

test("a protected folder this user cannot list, or that is gone, adds no members and blocks nothing else", async () => {
  const rules = [{ template: 'admin', resource: { entityType: 'Lark::Folder', entityId: 'fld_A' } }];
  for (const code of [1061004, 1061003]) {
    const g = guardWith(fakeAvp(rules), fakeLark({ failList: code }));
    assert.deepEqual(await g.decide('tokAlice', 'read', [d('doc_top'), f('fld_A')]), [true, false], `code ${code}`);
  }
});

test('concurrent calls share one walk of a protected folder', async () => {
  const rules = [{ template: 'admin', resource: { entityType: 'Lark::Folder', entityId: 'fld_A' } }];
  const lark = fakeLark();
  const g = guardWith(fakeAvp(rules), lark);
  await Promise.all([1, 2, 3].map(() => g.decide('tokAlice', 'read', [d('doc_A')])));
  assert.equal(lark.calls.filter((c) => c.includes('fld_A')).length, 1);
});

test('an error inside a policy denies, even though the baseline permit still matched', async () => {
  const g = guardWith(fakeAvp([], { errorsOn: 'doc_top' }), fakeLark());
  assert.deepEqual(await g.decide('tokAlice', 'read', [d('doc_top'), d('doc_A')]), [false, true]);
});

test('rules and listings are reused for a minute, then read again', async () => {
  let t = 0;
  const drive = { fld_A: [['doc_A', 'docx']] };
  const rules = [{ template: 'admin', resource: { entityType: 'Lark::Folder', entityId: 'fld_A' } }];
  const avp = fakeAvp(rules);
  const lark = fakeLark({ drive });
  const g = guardWith(avp, lark, { now: () => t });
  assert.deepEqual(await g.decide('tokAlice', 'read', [d('doc_moved')]), [true]);
  drive.fld_A.push(['doc_moved', 'docx']);           // moved into the protected folder
  t = 30_000;
  assert.deepEqual(await g.decide('tokAlice', 'read', [d('doc_moved')]), [true]);   // still the cached listing
  t = 61_000;
  assert.deepEqual(await g.decide('tokAlice', 'read', [d('doc_moved')]), [false]);  // covered within a minute
  assert.equal(lark.calls.filter((c) => c.includes('user_info')).length, 1);       // the owner is read once
});

test('lark_api reaches every Lark API; only a malformed method or path is refused', () => {
  for (const [m, p] of [['get', '/open-apis/im/v1/chats'], ['GET', '/open-apis/drive/v1/files'],
    ['POST', '/open-apis/docx/v1/documents'], ['PUT', '/open-apis/sheets/v2/spreadsheets/shtX/values/Sheet1!A1:B2']]) {
    assert.deepEqual(apiRoute(m, p), { method: m.toUpperCase(), path: p }, `${m} ${p}`);
  }
  // Positional arguments to lark-cli: a value shaped like a flag must not become one.
  for (const [m, p] of [['--as=bot', '/open-apis/im/v1/chats'], ['GET', '--as=bot'],
    ['GET', 'https://evil/open-apis/im/v1/x'], ['GET', '/open-apis/im/v1/chats --as bot'],
    // lark-cli would run the inner path, and Lark may fold these: the path checked must be the path run.
    ['GET', '/open-apis/im/v1/x/https://h/open-apis/docx/v1/documents/D/raw_content'],
    ['GET', '/open-apis//docx/v1/documents/D/raw_content'], ['GET', '/open-apis/im/../docx/v1/documents/D'],
    ['GET', '/open-apis/im/./x'], ['GET', '/open-apis/docx\\v1'],
    // A family segment Lark might read differently (`;` parameters, case) is not accepted at all.
    ['GET', `/open-apis/drive;x/v1/files/D/download`], ['GET', '/open-apis/DOCX/v1/documents/D/raw_content']]) {
    assert.ok(apiRoute(m, p).error, `${m} ${p} should be refused`);
  }
});

test('a lark_api call into Drive is matched to the documents and folders it touches', () => {
  const D = 'VgDzdeFAyoXgXyxfWL4jyJBlpzf', F = 'BwzCfk9ohlstzJdejkNjXZDJppg';
  const doc = { kind: 'document', token: D }, fld = { kind: 'folder', token: F };
  // Reading a document, by the token in the path.
  assert.deepEqual(classify('GET', `/open-apis/docx/v1/documents/${D}/raw_content`).items, [doc]);
  // Editing it: a write.
  const edit = classify('PATCH', `/open-apis/docx/v1/documents/${D}/blocks/blk1`, null, '{}');
  assert.equal(edit.action, 'write'); assert.deepEqual(edit.items, [doc]);
  // Creating a document inside a folder, by the folder in the body.
  assert.deepEqual(classify('POST', '/open-apis/docx/v1/documents', null, JSON.stringify({ folder_token: F, title: 't' })).items, [fld]);
  // A Drive file token can name either; both are checked.
  assert.deepEqual(classify('DELETE', `/open-apis/drive/v1/files/${F}`, JSON.stringify({ type: 'folder' })).items,
    [{ kind: 'document', token: F }, fld]);
  // Listing a folder, with the folder in a query string written into the path, and filtered on return.
  const list = classify('GET', `/open-apis/drive/v1/files?folder_token=${F}`);
  assert.deepEqual(list.items, [fld]); assert.equal(list.listing.key, 'files');
  // Batch metadata names its documents in request_docs.
  assert.deepEqual(classify('POST', '/open-apis/drive/v1/metas/batch_query', null,
    JSON.stringify({ request_docs: [{ doc_token: D, doc_type: 'docx' }] })).items, [doc, { kind: 'folder', token: D }]);
  // Search has no single resource, so its results are filtered instead.
  const search = classify('POST', '/open-apis/search/v2/doc_wiki/search', null, '{"query":"x"}');
  assert.deepEqual(search.items, []); assert.equal(search.listing.key, 'res_units');
});

test('outside the document families nothing is checked, and an unknown document API is refused', () => {
  assert.deepEqual(classify('GET', '/open-apis/im/v1/chats'), { document: false });
  assert.deepEqual(classify('POST', '/open-apis/calendar/v4/calendars', null, '{"summary":"x"}'), { document: false });
  assert.equal(classify('GET', '/open-apis/drive/v9/not-an-api').known, false);
  assert.equal(classify('POST', '/open-apis/docx/v1/documents', null, '{not json').unreadable, true);
});

test('a token hidden by encoding, case, arrays, nesting or another spelling is still found', () => {
  const D = 'VgDzdeFAyoXgXyxfWL4jyJBlpzf', F = 'BwzCfk9ohlstzJdejkNjXZDJppg';
  const doc = { kind: 'document', token: D }, fld = { kind: 'folder', token: F };
  // An encoded or upper-case family is still a document family, and matches no template.
  for (const p of [`/open-apis/%64ocx/v1/documents/${D}/raw_content`, `/open-apis/DOCX/v1/documents/${D}/raw_content`,
    `/open-apis/docx/v1/documents/%56gDzdeFAyoXgXyxfWL4jyJBlpzf/raw_content`]) {
    assert.equal(classify('GET', p).known, false, p);
  }
  // A query value given as an array is sent as repeated keys: every element is checked.
  const both = [doc, { kind: 'folder', token: D }];
  assert.deepEqual(classify('GET', '/open-apis/docs/v1/content', JSON.stringify({ doc_token: [D], doc_type: 'docx' })).items, both);
  assert.deepEqual(classify('GET', `/open-apis/docs/v1/content?doc_token=${D}&doc_token=${D}`).items, [...both, ...both]);
  // A type field is the caller's word: `folder` does not turn a document check into a folder-only one.
  assert.deepEqual(classify('GET', `/open-apis/docs/v1/content?doc_token=${D}&doc_type=folder`).items, both);
  // A list under a watched key: its tokens are checked and its objects searched, together.
  assert.deepEqual(classify('POST', '/open-apis/drive/v1/metas/batch_query', null,
    JSON.stringify({ folder_token: [F, { doc_token: D }] })).items, [fld, ...both]);
  // A watched field that holds something other than a token is refused, not skipped.
  assert.equal(classify('GET', '/open-apis/docs/v1/content', JSON.stringify({ doc_token: 'not a token' })).malformed, 'doc_token');
  assert.equal(classify('POST', '/open-apis/drive/v1/export_tasks', null, JSON.stringify({ token: { x: 1 }, type: 'docx' })).items.length, 0);
  assert.equal(classify('POST', '/open-apis/drive/v1/export_tasks', null, JSON.stringify({ token: 42 })).malformed, 'token');
  // Keys are matched without case or underscores, and anywhere in the body.
  assert.deepEqual(classify('POST', '/open-apis/drive/v1/export_tasks', null, JSON.stringify({ Token: D, type: 'docx' })).items, both);
  assert.deepEqual(classify('POST', '/open-apis/drive/v1/import_tasks', null,
    JSON.stringify({ file_token: 'boxXXXXXXXX1', type: 'docx', point: { mount_type: 1, mount_key: F } })).items,
  [{ kind: 'document', token: 'boxXXXXXXXX1' }, { kind: 'folder', token: 'boxXXXXXXXX1' }, fld]);
  // A bitable record's own column names are the user's, not Lark's.
  assert.deepEqual(classify('POST', `/open-apis/bitable/v1/apps/${D}/tables/tblX/records`, null,
    JSON.stringify({ fields: { token: 'any text', folder_token: 'x' } })).items, [doc]);
});

test('Sheets v2, legacy Docs and the Drive explorer are covered', () => {
  const S = 'shtcnXXXXXXXXXXXX1', D = 'doccnXXXXXXXXXXXX1', F = 'fldcnXXXXXXXXXXXX1';
  const cells = classify('GET', `/open-apis/sheets/v2/spreadsheets/${S}/values/0b12!A1:B2`);
  assert.equal(cells.action, 'read'); assert.deepEqual(cells.items, [{ kind: 'document', token: S }]);
  assert.equal(classify('PUT', `/open-apis/sheets/v2/spreadsheets/${S}/values`, null, '{"valueRange":{"range":"0b12!A1","values":[["x"]]}}').action, 'write');
  assert.deepEqual(classify('GET', `/open-apis/doc/v2/${D}/raw_content`).items, [{ kind: 'document', token: D }]);
  assert.deepEqual(classify('GET', `/open-apis/doc/v2/meta/${D}`).items, [{ kind: 'document', token: D }]);
  assert.deepEqual(classify('POST', '/open-apis/doc/v2/create', null, JSON.stringify({ FolderToken: F })).items, [{ kind: 'folder', token: F }]);
  const kids = classify('GET', `/open-apis/drive/explorer/v2/folder/${F}/children`);
  assert.deepEqual(kids.items, [{ kind: 'folder', token: F }]); assert.equal(kids.listing.key, 'children');
  assert.deepEqual(classify('POST', `/open-apis/drive/explorer/v2/file/copy/files/${D}`, null, JSON.stringify({ dstFolderToken: F })).items,
    [{ kind: 'document', token: D }, { kind: 'folder', token: D }, { kind: 'folder', token: F }]);
  // A literal segment wins over a parameter: `query` is not a sheet id.
  assert.deepEqual(classify('GET', `/open-apis/sheets/v3/spreadsheets/${S}/sheets/query`).items, [{ kind: 'document', token: S }]);
});

test('calls that need no token run, and an encoded path is refused with a reason', () => {
  assert.deepEqual(classify('GET', '/open-apis/search/v1/user?query=alice').items, []);
  assert.equal(classify('GET', '/open-apis/sheets/v2/spreadsheets/shtcnXXXXXXXXXXXX1/values/0b12%21A1%3AB2').encoded, true);
});
