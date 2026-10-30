// What a raw `lark_api` call may carry, and which Drive documents and folders it touches. Every
// Lark API stays callable; a call into a document family (doc_routes.json, generated from the
// official SDK) is checked against the protection rules before it runs, and listings it returns
// are filtered. Kept apart from server.js, which starts listening when loaded, so it can be tested.
'use strict';

const DOC = require('./doc_routes.json');

const TOKEN_RE = /^[A-Za-z0-9]{8,64}$/;
const METHODS = new Set(['GET', 'POST', 'PUT', 'PATCH', 'DELETE']);
const FAMILIES = new Set(DOC.families);
// Keys compare without case or underscores: `folder_token`, `folderToken` and `FolderToken` are
// one field to Lark's JSON decoding, and so to the guard.
const norm = (k) => String(k).toLowerCase().replace(/_/g, '');
const PARAM = /:[A-Za-z_]+/g;
// A template ending in `/*` covers everything beneath it, for any method.
const ROUTES = DOC.routes.map((r) => {
  const rest = r.template.endsWith('/*');
  const t = rest ? r.template.slice(0, -2) : r.template;
  const names = (t.match(PARAM) || []).map((n) => norm(n.slice(1)));
  return {
    method: r.method, template: r.template, names,
    re: new RegExp(`^${t.replace(PARAM, '([^/]+)')}${rest ? '(?:/.*)?' : ''}$`),
    // Literal segments win over parameters, and an exact template over a `/*` one.
    rank: (rest ? 100 : 0) + names.length,
  };
});

// Where a resource token sits: a path parameter, or a field anywhere in the query or body.
// "either" is a token that can name a document or a folder (a Drive file token), so both are
// checked; a type field beside it is the caller's word, so it narrows nothing.
const PATH_PARAMS = {
  documentid: 'document', filetoken: 'either', token: 'either', spreadsheettoken: 'document',
  apptoken: 'document', minutetoken: 'document', whiteboardid: 'document', doctoken: 'document',
  foldertoken: 'folder',
};
const FIELDS = {
  foldertoken: 'folder', parenttoken: 'folder', mountkey: 'folder', dstfoldertoken: 'folder',
  parentnode: 'either', objtoken: 'document', documentid: 'document', spreadsheettoken: 'document',
  refertoken: 'document', doctoken: 'either', docstoken: 'either', filetoken: 'either', token: 'either',
};
// A bitable record's `fields` is keyed by the user's own column names, not by Lark.
const USER_KEYED = new Set(['fields']);
// Listings whose entries are filtered: where they sit in `data` (an array, or a map keyed by
// token), and each entry's token and type.
const LISTINGS = {
  'GET /open-apis/drive/v1/files': { key: 'files', token: (e) => e.token, type: (e) => e.type },
  'GET /open-apis/drive/explorer/v2/folder/:folder_token/children': { key: 'children', token: (e) => e.token, type: (e) => e.type },
  'POST /open-apis/suite/docs-api/search/object': { key: 'docs_entities', token: (e) => e.docs_token, type: (e) => e.docs_type },
  'POST /open-apis/drive/v1/metas/batch_query': { key: 'metas', token: (e) => e.doc_token, type: (e) => e.doc_type },
  'POST /open-apis/search/v2/doc_wiki/search': {
    key: 'res_units', token: (e) => e.result_meta && e.result_meta.token,
    type: (e) => String((e.result_meta && e.result_meta.doc_types) || '').toLowerCase(),
  },
};

// lark-cli rewrites a path holding `http(s)://host/open-apis/...` to that inner path, and Lark may
// fold `//`, dot segments or backslashes; any of them would make the path checked differ from
// the path run, so none is accepted.
function apiRoute(method, path) {
  const m = String(method || '').toUpperCase();
  const p = String(path || '');
  if (!METHODS.has(m)) return { error: 'method must be one of GET, POST, PUT, PATCH, DELETE' };
  if (!p.startsWith('/open-apis/') || /[\s\\#]|:\/\/|\/\/|\/\.\.?(\/|\?|$)/.test(p)
      || !/^[a-z0-9_]+$/.test(p.split(/[/?]/)[2] || '')) {
    return { error: 'path must be a plain /open-apis/... path' };
  }
  return { method: m, path: p };
}

class Malformed extends Error {}

const items = (kind, token) => (kind === 'either'
  ? [{ kind: 'document', token }, { kind: 'folder', token }] : [{ kind, token }]);

// A watched field must hold a token (or nothing): anything else could not be checked.
function take(kind, key, v, out) {
  if (v === '' || v === null || v === undefined) return;
  if (Array.isArray(v)) { v.forEach((x) => take(kind, key, x, out)); return; }
  if (typeof v !== 'string' || !TOKEN_RE.test(v)) throw new Malformed(key);
  out.push(...items(kind, v));
}

function fieldsOf(obj, out, depth = 0) {
  if (!obj || typeof obj !== 'object') return;
  if (depth > 32) throw new Malformed('nesting');
  if (Array.isArray(obj)) { obj.forEach((x) => fieldsOf(x, out, depth + 1)); return; }
  for (const [k, v] of Object.entries(obj)) {
    const kind = FIELDS[norm(k)];
    if (!kind) {
      if (!(USER_KEYED.has(norm(k)) && v && typeof v === 'object' && !Array.isArray(v))) fieldsOf(v, out, depth + 1);
    } else if (Array.isArray(v)) {
      // Each element: a token is checked, an object is searched in turn.
      v.forEach((x) => (x && typeof x === 'object' ? fieldsOf(x, out, depth + 1) : take(kind, k, x, out)));
    } else if (v && typeof v === 'object') {
      fieldsOf(v, out, depth + 1);
    } else {
      take(kind, k, v, out);
    }
  }
}

// lark-cli sends repeated query keys for an array value, so a key can carry several tokens.
function queryOf(query) {
  const q = {};
  for (const [k, v] of new URLSearchParams(query)) (q[k] = q[k] || []).push(v);
  return q;
}

// → { document: false } for an API outside the document families; otherwise { document: true,
// known, unreadable, malformed, action, items, listing }. A query string in `path` is read too.
function classify(method, path, params, data) {
  const [bare, query] = String(path).split('?');
  let family;
  try { family = decodeURIComponent(bare.split('/')[2]).toLowerCase(); } catch { family = ''; }
  // An encoded or differently cased family still counts as one; it then matches no template.
  if (!FAMILIES.has(family)) return /%/.test(bare.split('/')[2]) ? { document: true, known: false } : { document: false };
  if (/%/.test(bare)) return { document: true, known: false, encoded: true };
  const route = ROUTES.filter((r) => (r.method === method || r.method === '*') && r.re.test(bare))
    .sort((a, b) => a.rank - b.rank)[0];
  if (!route) return { document: true, known: false };
  const found = [];
  try {
    const values = bare.match(route.re).slice(1);
    route.names.forEach((n, i) => { if (PATH_PARAMS[n]) take(PATH_PARAMS[n], n, values[i], found); });
    let parsed;
    try {
      parsed = [query && queryOf(query), params && JSON.parse(params), data && JSON.parse(data)];
    } catch {
      return { document: true, known: true, unreadable: true };
    }
    parsed.forEach((o) => fieldsOf(o, found));
  } catch (e) {
    if (e instanceof Malformed) return { document: true, known: true, malformed: e.message };
    throw e;
  }
  const listing = LISTINGS[`${method} ${route.template}`];
  return { document: true, known: true, action: method === 'GET' ? 'read' : 'write', items: found, listing };
}

module.exports = { TOKEN_RE, apiRoute, classify };
