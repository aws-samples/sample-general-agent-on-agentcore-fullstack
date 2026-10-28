// The guards are the point of this server, so they get tested. Unlike the lark-cli
// MCP server (a thin passthrough), this one decides whether a write to Lark happens
// at all — and Lark's approve/reject accept no user token, so nothing downstream
// will second-guess it. Node's built-in runner, no new dependencies:
//   node --test mcp-servers/approval/
//
// server.js is CommonJS and starts listening on import, so the guard functions are
// extracted and evaluated in isolation rather than imported.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

function loadGuards(env = {}, fetchImpl = undefined) {
  const src = readFileSync(new URL('../../mcp-servers/approval/server.js', import.meta.url), 'utf8');
  // Anchored on declarations, not on comment text — an earlier version keyed off a
  // comment and broke the moment that comment was reworded.
  const start = src.indexOf('const AGENT_DECIDE_MAX_AMOUNT');
  const end = src.indexOf('const TOOLS = [');
  assert.ok(start > 0 && end > start, 'guard block not found — did server.js move?');
  const fn = new Function('process', 'fetch', 'API', `${src.slice(start, end)}
    return { checkAutoDecisionAllowed, requireConsentOnRecord, consentNote, policyNote, formAmount, readInstance, isParticipant, AGENT_DECIDE_MAX_AMOUNT, AGENT_DECIDE_APPROVAL_CODES };`);
  return fn({ env }, fetchImpl, 'https://open.larksuite.com');
}

test('fails closed: no configured approval codes means nothing is automated', () => {
  const g = loadGuards({});                       // AGENT_DECIDE_APPROVAL_CODES unset
  const refusal = g.checkAutoDecisionAllowed({ approval_code: 'ANY', amount: 1 });
  assert.match(refusal, /not in AGENT_DECIDE_APPROVAL_CODES/);
});

test('only allow-listed approval definitions are automated', () => {
  const g = loadGuards({ AGENT_DECIDE_APPROVAL_CODES: 'EXPENSE_V1,LEAVE_V1' });
  assert.equal(g.checkAutoDecisionAllowed({ approval_code: 'EXPENSE_V1', amount: 10 }), null);
  assert.match(g.checkAutoDecisionAllowed({ approval_code: 'PAYROLL_V1', amount: 10 }),
               /not in AGENT_DECIDE_APPROVAL_CODES/);
});

test('amount above the limit is refused, at the limit is allowed', () => {
  const g = loadGuards({ AGENT_DECIDE_APPROVAL_CODES: 'EXPENSE_V1', AGENT_DECIDE_MAX_AMOUNT: '1000' });
  assert.equal(g.checkAutoDecisionAllowed({ approval_code: 'EXPENSE_V1', amount: 1000 }), null);
  assert.match(g.checkAutoDecisionAllowed({ approval_code: 'EXPENSE_V1', amount: 1000.01 }),
               /exceeds the automatic limit/);
});

test('a kill switch exists: 0 disables automation even for allow-listed codes', () => {
  const g = loadGuards({ AGENT_DECIDE_APPROVAL_CODES: 'EXPENSE_V1', AGENT_DECIDE_MAX_AMOUNT: '0' });
  assert.match(g.checkAutoDecisionAllowed({ approval_code: 'EXPENSE_V1', amount: 1 }),
               /disabled/);
});

test('an amount the form carries but cannot be read is refused', () => {
  const g = loadGuards({ AGENT_DECIDE_APPROVAL_CODES: 'EXPENSE_V1', AGENT_DECIDE_MAX_AMOUNT: '1000' });
  assert.match(g.checkAutoDecisionAllowed({ approval_code: 'EXPENSE_V1', amount: null, unreadable: true }),
               /could not be read/);
  // A form with no amount at all (a trip request) is limited by the allow-list alone.
  assert.equal(g.checkAutoDecisionAllowed({ approval_code: 'EXPENSE_V1', amount: null }), null);
});

test('the amount is summed from the form, rows of a detail list included', () => {
  // Shape measured on a real expense instance: the amount sits in each row of a
  // fieldList, and the list repeats its total in `ext`, which must not count twice.
  const g = loadGuards({});
  const form = [
    { type: 'textarea', value: 'travel to guangzhou' },
    { type: 'fieldList', value: [[{ type: 'amount', value: 375 }], [{ type: 'amount', value: '900.5' }]],
      ext: [{ type: 'amount', value: '1275.50' }] },
  ];
  assert.deepEqual(g.formAmount(form), { amount: 1275.5, unreadable: false });
  assert.deepEqual(g.formAmount([{ type: 'tripGroup', value: {} }]), { amount: null, unreadable: false });
  assert.equal(g.formAmount([{ type: 'amount', value: 'lots' }]).unreadable, true);
  // Neither a malformed figure nor a negative row may lower what is checked.
  for (const value of ['1,234.00', '12abc', -8500, '']) {
    assert.equal(g.formAmount([{ type: 'amount', value: 9000 }, { type: 'amount', value }]).unreadable, true, String(value));
  }
});

function instanceFetch(instance) {
  return async () => ({ status: 200, json: async () => ({ code: 0, data: instance }) });
}

const INSTANCE = {
  approval_code: 'EXPENSE_V1',
  form: JSON.stringify([{ type: 'amount', value: 5000 }]),
  task_list: [{ id: 't1', open_id: 'ou_alice', status: 'PENDING' }],
};

test("the guard reads the instance, so the model's arguments cannot lower the amount", async () => {
  // The bypass this replaced: approve without `amount` sailed past the limit, because the
  // only figure checked was the one the model chose to pass.
  const g = loadGuards({ AGENT_DECIDE_APPROVAL_CODES: 'EXPENSE_V1', AGENT_DECIDE_MAX_AMOUNT: '1000' },
                       instanceFetch(INSTANCE));
  const facts = await g.readInstance('i1', 't1', 'ou_alice', async () => 'tok');
  assert.equal(facts.approval_code, 'EXPENSE_V1');
  assert.match(g.checkAutoDecisionAllowed(facts), /exceeds the automatic limit/);
});

test('a task that is not a pending task of this approver on this instance is refused', async () => {
  const g = loadGuards({}, instanceFetch(INSTANCE));
  for (const [task, user] of [['t2', 'ou_alice'], ['t1', 'ou_bob']]) {
    assert.match((await g.readInstance('i1', task, user, async () => 'tok')).error, /not a pending task/);
  }
  const done = loadGuards({}, instanceFetch({ ...INSTANCE, task_list: [{ id: 't1', open_id: 'ou_alice', status: 'APPROVED' }] }));
  assert.match((await done.readInstance('i1', 't1', 'ou_alice', async () => 'tok')).error, /not a pending task/);
});

test('an instance that cannot be read is refused, not waved through', async () => {
  const g = loadGuards({}, async () => ({ status: 200, json: async () => ({ code: 1, msg: 'no permission' }) }));
  assert.match((await g.readInstance('i1', 't1', 'ou_alice', async () => 'tok')).error, /could not read instance/);
  const down = loadGuards({}, async () => { throw new Error('ECONNRESET'); });
  assert.match((await down.readInstance('i1', 't1', 'ou_alice', async () => 'tok')).error, /ECONNRESET/);
});

test('every tool that moves a task is guarded, transfer included', () => {
  const src = readFileSync(new URL('../../mcp-servers/approval/server.js', import.meta.url), 'utf8');
  for (const name of ['approval_approve', 'approval_reject', 'approval_transfer']) {
    const block = src.slice(src.indexOf(`name: '${name}'`), src.indexOf('req:', src.indexOf(`name: '${name}'`)));
    assert.match(block, /guarded: true/, name);
  }
});

test('refuses to decide when no authorization is on hand', async () => {
  // Lark would allow this — approve/reject take only a tenant token, so the API never
  // checks whether the approver agreed. Refusing is this server's own rule.
  const g = loadGuards({});
  const refusal = await g.requireConsentOnRecord('', 'ou_alice');
  assert.match(refusal, /no on-record authorization/);
  assert.match(refusal, /the limit is ours/);          // says who is enforcing it
});

test("refuses when the authorization belongs to someone other than the approver", async () => {
  // The agent holds a token for every user who ever consented, so "a token exists" and
  // "this approver consented" are different questions. Deciding in Alice's name using
  // Bob's grant is the impersonation this closes.
  const fake = async () => ({ json: async () => ({ data: { open_id: 'ou_bob_1234567890' } }) });
  const g = loadGuards({}, fake);
  const refusal = await g.requireConsentOnRecord('bobs-token', 'ou_alice');
  assert.match(refusal, /belongs to ou_bob/);
  assert.match(refusal, /that approver's own grant/);
});

test('allows the decision when the authorization is the approver\'s own', async () => {
  const fake = async () => ({ json: async () => ({ data: { open_id: 'ou_alice' } }) });
  const g = loadGuards({}, fake);
  assert.equal(await g.requireConsentOnRecord('alices-token', 'ou_alice'), null);
});

test('refuses when the identity behind the token cannot be established', async () => {
  // Fail closed: an unverifiable token must not pass as "probably the right person".
  const bad = async () => ({ json: async () => ({ code: 99991663, msg: 'invalid token' }) });
  const g = loadGuards({}, bad);
  assert.match(await g.requireConsentOnRecord('junk', 'ou_alice'), /could not establish/);

  const boom = async () => { throw new Error('network down'); };
  const g2 = loadGuards({}, boom);
  assert.match(await g2.requireConsentOnRecord('t', 'ou_alice'), /could not verify/);
});

test('the advertised policy states the real limits, so the model need not guess', () => {
  // A model asked whether a case is "in scope", without being told the scope, falls back
  // on its context: one refused a ¥375 reimbursement the allow-list permitted, citing an
  // authorization boundary that did not exist. The limits still bind in code — this only
  // stops the model inventing a stricter, invisible one.
  const g = loadGuards({ AGENT_DECIDE_APPROVAL_CODES: 'AC1, AC2', AGENT_DECIDE_MAX_AMOUNT: '500' });
  const note = g.policyNote();
  assert.match(note, /AC1, AC2/);
  assert.match(note, /500/);
  assert.match(note, /Do not refuse on your own judgement of scope/);
  assert.match(note, /enforced in this server, not by you/);
});

test('the advertised policy does not claim permission when nothing is allowed', () => {
  const off = loadGuards({});
  assert.match(off.policyNote(), /none — every automatic decision is refused/);
  const killed = loadGuards({ AGENT_DECIDE_APPROVAL_CODES: 'AC1', AGENT_DECIDE_MAX_AMOUNT: '0' });
  assert.match(killed.policyNote(), /disabled outright/);
});

test('the consent note distinguishes an on-record grant from app authority alone', () => {
  const g = loadGuards({});
  assert.match(g.consentNote(true), /on-record 3LO grant/);
  assert.match(g.consentNote(false), /NO on-record grant/);
  // The wording must stay explicit: the record is what tells a reader afterwards
  // whether anyone had agreed to be acted for.
  assert.notEqual(g.consentNote(true), g.consentNote(false));
});

test("reads are limited to what the caller could see in Lark: their queue, their instances", () => {
  // The app token sees every queue and instance in the tenant; asked for someone else's,
  // the model would otherwise read it.
  const g = loadGuards({});
  const instance = { open_id: 'ou_starter', task_list: [{ open_id: 'ou_approver' }], cc_list: [{ open_id: 'ou_cc' }] };
  for (const who of ['ou_starter', 'ou_approver', 'ou_cc']) assert.equal(g.isParticipant(instance, who), true, who);
  assert.equal(g.isParticipant(instance, 'ou_stranger'), false);
  assert.equal(g.isParticipant(undefined, 'ou_starter'), false);
  const src = readFileSync(new URL('../../mcp-servers/approval/server.js', import.meta.url), 'utf8');
  const block = (name) => src.slice(src.indexOf(`name: '${name}'`), src.indexOf('req:', src.indexOf(`name: '${name}'`)));
  assert.match(block('approval_list_pending'), /ownQueue: true/);
  assert.match(block('approval_get_instance'), /participantsOnly: true/);
});
