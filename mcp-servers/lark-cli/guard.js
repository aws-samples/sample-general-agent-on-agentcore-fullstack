// Which Lark documents and folders this user's agent may touch, decided by Cedar policies in
// Amazon Verified Permissions. Lark grants a user token everything the user can see, so the
// restriction is enforced here, before every call; see docs/architecture.md.
//
// Fails closed: anything that cannot be decided — AVP unreachable, an error inside a policy, a
// folder listing that fails for a passing reason, a token whose owner cannot be read — is a
// refusal, reported as unchecked rather than as protected. A protected folder this user may not
// list, or that no longer exists, holds nothing they can reach through it: it adds no members.
'use strict';
const crypto = require('crypto');

const RULES_TTL_MS = 60_000;      // a new rule, or a document moved into a protected folder, holds within a minute
const OWNER_TTL_MS = 10 * 60_000; // a token's owner never changes; well inside the token's own lifetime
const MAX_ENTRIES = 5000;         // per protected folder: past this the walk is refused, not half-done
const BATCH = 30;                 // BatchIsAuthorized's limit per call
const PARALLEL = 4;               // folder listings in flight per walk
const CACHE_CAP = 2000;           // entries per cache before it is cleared
// Lark's answers for a folder this token may not list, or that is gone. Anything else from a
// listing (rate limits, timeouts) is passing, and leaves the decision open.
const NOT_REACHABLE = new Set([1061003, 1061004, 1061007, 1062507]);

class Undecided extends Error {}

const doc = (token) => ({ entityType: 'Lark::Document', entityId: token });
const folder = (token) => ({ entityType: 'Lark::Folder', entityId: token });

// `avp` is a VerifiedPermissions client (or a fake): { send(command) }. `commands` carries the
// SDK's command classes so this module does not import the SDK. `lark(args, token)` runs a
// lark-cli call and resolves to its parsed JSON.
function createGuard({ avp, commands, lark, storeId, adminTemplateId, userTemplateId, appId,
  now = Date.now, log = console.log }) {
  const owners = new Map();
  const rules = new Map();
  const walks = new Map();
  const configured = Boolean(storeId && adminTemplateId && userTemplateId && appId);

  // Concurrent callers share one computation; a failure is dropped, not cached.
  function cached(map, key, ttl, compute) {
    const hit = map.get(key);
    if (hit && hit.exp > now()) return hit.value;
    if (map.size > CACHE_CAP) map.clear();
    const value = compute();
    map.set(key, { value, exp: now() + ttl });
    value.catch(() => { if (map.get(key) && map.get(key).value === value) map.delete(key); });
    return value;
  }

  const hash = (token) => crypto.createHash('sha256').update(token).digest('hex');

  // The principal is the Lark account the token belongs to, read from Lark with the token
  // itself: never a value the caller states. Scoped to the app, because an open_id is per app.
  async function principalFor(token) {
    const openId = await cached(owners, hash(token), OWNER_TTL_MS, async () => {
      const r = await lark(['api', 'GET', '/open-apis/authen/v1/user_info'], token);
      const id = r && r.ok && r.data && r.data.open_id;
      if (!id) throw new Undecided('the token owner could not be read');
      return id;
    });
    return { entityType: 'Lark::User', entityId: `${appId}/${openId}` };
  }

  async function linked(filter) {
    const out = [];
    let nextToken;
    do {
      const page = await avp.send(new commands.ListPoliciesCommand({
        policyStoreId: storeId, filter, ...(nextToken ? { nextToken } : {}) }));
      out.push(...(page.policies || []));
      nextToken = page.nextToken;
    } while (nextToken);
    return out;
  }

  // Folders under protection for this principal: the admins' rules plus its own. Document
  // rules need nothing resolved — a document is matched by its token.
  async function protectedFolders(principal) {
    return cached(rules, principal.entityId, RULES_TTL_MS, async () => {
      const policies = [
        ...await linked({ policyTemplateId: adminTemplateId }),
        ...await linked({ policyTemplateId: userTemplateId, principal: { identifier: principal } }),
      ];
      const tokens = new Set();
      for (const p of policies) {
        const r = p.resource || (p.definition && p.definition.templateLinked && p.definition.templateLinked.resource);
        if (r && r.entityType === 'Lark::Folder') tokens.add(r.entityId);
      }
      return [...tokens];
    });
  }

  // Lark has no "which folder is this document in" API, so each protected folder is listed
  // downward with the caller's own token. Result: entity id → the folders above it, nearest first.
  async function walk(root, token) {
    return cached(walks, `${root}|${hash(token)}`, RULES_TTL_MS, async () => {
      const above = new Map([[root, []]]);
      let level = [root];
      let seen = 0;
      const listAll = async (f) => {
        const children = [];
        let pageToken = '';
        do {
          const params = { folder_token: f, page_size: 200, ...(pageToken ? { page_token: pageToken } : {}) };
          const r = await lark(['api', 'GET', '/open-apis/drive/v1/files', '--params', JSON.stringify(params)], token);
          if (!r || !r.ok) {
            const code = r && r.error && r.error.code;
            if (NOT_REACHABLE.has(code)) {
              log(`protection: folder ${f} not reachable with this token (code ${code}); it adds no members`);
              return children;
            }
            throw new Undecided(`folder ${f} could not be listed (code ${code || 'none'})`);
          }
          children.push(...((r.data && r.data.files) || []));
          pageToken = r.data && r.data.has_more ? r.data.next_page_token : '';
        } while (pageToken);
        return children;
      };
      while (level.length) {
        const next = [];
        for (let i = 0; i < level.length; i += PARALLEL) {
          const batch = level.slice(i, i + PARALLEL);
          const lists = await Promise.all(batch.map(listAll));
          lists.forEach((children, n) => {
            for (const e of children) {
              if (++seen > MAX_ENTRIES) throw new Undecided(`folder ${root} holds more than ${MAX_ENTRIES} entries`);
              above.set(e.token, [batch[n], ...above.get(batch[n])]);
              if (e.type === 'folder') next.push(e.token);
            }
          });
        }
        level = next;
      }
      return above;
    });
  }

  // Decide `action` on each item ({kind: 'document'|'folder', token}), in order: true (allowed),
  // false (protected) or null (could not be checked). Any failure leaves every item unchecked:
  // an unanswered question is never an allow.
  async function decide(token, action, items) {
    if (!items.length) return [];
    try {
      if (!configured) throw new Undecided('resource protection is not configured');
      const principal = await principalFor(token);
      const parents = new Map();             // entity id → Set of folder tokens above it
      for (const root of await protectedFolders(principal)) {
        for (const [id, chain] of await walk(root, token)) {
          if (!parents.has(id)) parents.set(id, new Set());
          chain.forEach((f) => parents.get(id).add(f));
        }
      }
      const entity = (it) => (it.kind === 'folder' ? folder(it.token) : doc(it.token));
      const entityList = new Map();
      const add = (e) => {
        const key = `${e.entityType}/${e.entityId}`;
        if (entityList.has(key)) return;
        entityList.set(key, { identifier: e, parents: [...(parents.get(e.entityId) || [])].map(folder) });
        for (const f of parents.get(e.entityId) || []) add(folder(f));
      };
      items.forEach((it) => add(entity(it)));
      const allowed = new Map();             // "type/id" → decision, matched on the echoed request
      for (let i = 0; i < items.length; i += BATCH) {
        const chunk = items.slice(i, i + BATCH);
        const r = await avp.send(new commands.BatchIsAuthorizedCommand({
          policyStoreId: storeId,
          entities: { entityList: [...entityList.values()] },
          requests: chunk.map((it) => ({
            principal, action: { actionType: 'Lark::Action', actionId: action }, resource: entity(it) })),
        }));
        // Cedar skips a policy that errors and lets the rest decide, so an error inside a
        // forbid would otherwise leave the baseline permit standing: an error is a deny.
        for (const res of r.results || []) {
          const q = (res.request && res.request.resource) || {};
          allowed.set(`${q.entityType}/${q.entityId}`, res.decision === 'ALLOW' && !(res.errors || []).length);
        }
      }
      const out = items.map((it) => {
        const e = entity(it);
        const key = `${e.entityType}/${e.entityId}`;
        if (!allowed.has(key)) throw new Undecided(`AVP did not answer for ${key}`);
        return allowed.get(key);
      });
      items.forEach((it, n) => { if (!out[n]) log(`protection: denied ${action} ${it.kind} ${it.token} for ${principal.entityId}`); });
      return out;
    } catch (e) {
      log(`protection: unchecked (${e.message}); refusing ${items.length} item(s)`);
      return items.map(() => null);
    }
  }

  return { decide, configured };
}

module.exports = { createGuard, Undecided };
