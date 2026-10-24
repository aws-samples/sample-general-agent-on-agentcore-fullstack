# Deploying this sample

The README has the short version. This page is the detail: every step, every optional feature, what teardown costs you, and what it bills while it runs.

## Steps

Prereqs: `uv` (Docker only with `AGENT_BUILD=local`), the AgentCore CLI (`npm i -g @aws/agentcore`), and AWS credentials. The deployment target lives in `.env` (`PROFILE`, `REGION`, `MODEL_ID`); command-line env vars override it (`REGION=... ./deploy.sh`). Resources deploy under the `agentcore-fullstack` prefix. The first deploy into a region runs `cdk bootstrap` automatically.

```bash
cp .env.example .env          # deployment target (PROFILE/REGION/MODEL_ID) + Lark appId/appSecret/encryptKey/token + your open_id
./deploy.sh                   # everything, in order — ends by printing the two URLs to register in Lark
```

That's the whole deploy. It runs the steps in dependency order and nothing stops to ask you anything: the two values you must paste into the Lark console don't block the deploy, they only gate the bot at runtime, so they're printed together at the end (`./deploy.sh urls` reprints them).

Individual steps, for iterating — each is idempotent, so re-running any of them is safe:

| Step | What |
|---|---|
| `./deploy.sh base` | CDK stacks (security, agentcore, router, shim, gateway, observability, plus storage when `FILES_STORAGE=true`) |
| `./deploy.sh code` | the Code Interpreter generated code runs in — skipped unless `FILES_STORAGE=true` |
| `./deploy.sh webui` | the web chat page on S3 + CloudFront — skipped unless `WEBUI=true` |
| `./deploy.sh a2a` | the A2A Runtime (same image, `SERVER_MODE=a2a`) — then `scripts/a2a-demo.sh` drives it as a peer would |
| `./deploy.sh mcp` | build every MCP server under `mcp-servers/` (CodeBuild ARM64) + create/update a Runtime each. `./deploy.sh mcp approval` for just one |
| `./deploy.sh 3lo` | workload identity + the `agentcore-fullstack-3lo` OAuth credential provider |
| `./deploy.sh gateway` | Web Search gateway in us-east-1 — skipped unless `WEB_SEARCH=true` |
| `./deploy.sh runtime` | point the agent Runtime at the image `base` published — agent code changes need `base` first |
| `./deploy.sh lark` | seed Lark credentials to Secrets Manager + allowlist your `open_id` |
| `./deploy.sh approvals` | subscribe to Lark approval events for each `AGENT_DECIDE_APPROVAL_CODES` definition — no-op when that is empty. Ticking the event in the console is **not** sufficient; Lark delivers approval events only for definitions also subscribed through the API |

Order matters in one place: `3lo` and `gateway` precede `runtime`, because the agent Runtime is created with the provider name and the gateway URL baked into its environment. `deploy.sh` handles that; the underlying implementations are in `scripts/`.

`setup-3lo.sh` registers the OAuth credential provider (Lark behind the RFC-6749 shim) plus the agent's workload identity, and prints the provider `callbackUrl` — register that in the Lark console ([step 4](docs/console-setup.md#lark-console-setup)) before the first 3LO consent.

Per-deployment ids (runtime/gateway) go to `.cdk-state.json` (gitignored), so `cdk.json` stays free of environment state.

Optional web chat: set `WEBUI=true` in `.env`, then register the printed domain on the Lark app ([step 5](docs/console-setup.md#lark-console-setup)). CloudFront only assigns the domain at deploy time and the router must allow that exact origin to trade an h5 code for a JWT, so `./deploy.sh webui` writes it to `.cdk-state.json` and re-deploys the router itself — the same route the file system id takes. Off by default: without the Lark-side registration the page loads but can establish nobody.

Optional web search: set `WEB_SEARCH=true` in `.env` before deploying (or run `./deploy.sh gateway` then `./deploy.sh runtime`). That provisions a Gateway fronting AgentCore's built-in Web Search connector — **in us-east-1, the only region offering it**, so the agent calls it cross-region. Search carries no end-user identity, so it uses `GATEWAY_IAM_ROLE` and never touches the per-user 3LO path the Lark tools have to avoid. Off by default; the agent just runs without the tool.

### Tear down

```bash
scripts/destroy.sh            # delete everything deploy.sh created (asks for confirmation)
# or: scripts/destroy.sh --yes  # skip the prompt
```

Deletes in dependency order — gateway targets → gateways (including the Web Search one in us-east-1) → both CLI-created Runtimes → the CDK stacks → the OAuth credential provider and workload identity. Idempotent: re-running skips already-gone resources.

Two consequences worth knowing: deleting the provider **purges every user's vaulted token**, so everyone consents again after a redeploy — and the new provider gets a **new `callbackUrl`** that must be registered in the Lark console (`scripts/setup-3lo.sh` prints it). Your Lark console app config is otherwise untouched; re-seed credentials from `.env` via `scripts/setup-lark.sh`.

## Letting more people in

The bot answers only allowlisted users; `./deploy.sh lark` adds you and nobody else. An unlisted user who messages the bot is told their own id, which is the easiest way to collect one:

```bash
PROFILE=... REGION=... scripts/manage-allowlist.sh add lark:ou_...
scripts/manage-allowlist.sh list
scripts/manage-allowlist.sh remove lark:ou_...
```

The allowlist gates *conversations*. It also gates whether an approval event is acted on at all: an approver who isn't listed is left alone silently. Note the asymmetry — deciding for someone needs their own 3LO grant, but DMing them does not (that runs on the app's token), so the allowlist is the only thing standing between an approval event and a stranger's chat window.

## Cost

This deploys billable AWS resources. All the always-on pieces are consumption- or per-unit-priced (no fixed reservation), so an idle single-user demo in us-west-2 is on the order of a couple USD/month before model usage; the variable cost is dominated by the agent's Bedrock calls. Verify current rates on the AWS pricing pages — figures below are as researched, not a quote.

- **Bedrock model invocations** — the main usage-sensitive line; priced per input/output token on the model in `MODEL_ID`. A chatty demo is cents-to-dollars; a load test is not.
- **AgentCore Runtime ×2–3** — the agent and the lark-cli MCP server always, plus the approval MCP server if `AGENT_DECIDE_APPROVAL_CODES` is set. Each is metered per-second: CPU (`~$0.0895/vCPU-hour`) only during active processing, memory (`~$0.00945/GB-hour`) continuously while the microVM is alive. So each extra MCP server adds idle memory-time even when nothing calls it — which is why the approval one is gated on that variable rather than always deployed, and why grouping tools by trust boundary has a running cost.
- **AgentCore Identity Token Vault (3LO)** — stores/refreshes/injects each user's Lark token natively. No separate per-user Secrets Manager charge (unlike the interceptor variant).
- **DynamoDB + S3 (checkpoints)** — on-demand writes, one item per superstep, with state over ~350 KB spilled to S3. Table TTL expires old threads; nothing is billed for retention beyond storage.

- **AgentCore Memory** — long-term records only. Billed per event *written* (`~$0.25 per 1,000`), per record stored per month, and per *retrieval* (`~$0.50 per 1,000`) — retrieval is the expensive one, so it is queried on demand rather than every turn.
- **Lambda + API Gateway** — router (webhook) + shim (OAuth RFC-6749 façade, a backend web service); effectively free at demo volume.
- **Secrets Manager** — `$0.40/secret/month` each, and only two static secrets: the Lark credentials (`{prefix}/channels/lark`) and the Cognito password salt. No dynamic per-user secrets.
- **Web search (optional)** — only when `WEB_SEARCH=true`: an AgentCore Gateway plus per-query connector charges. The gateway sits in us-east-1, so its traffic is cross-region.
- **Code execution (optional, `FILES_STORAGE=true`)** — **no NAT.** A mount needs a VPC, but only the Code Interpreter session goes in there, and it only has to reach S3, which a free gateway endpoint does. So the fixed cost is a VPC (free), an S3 gateway endpoint (free) and mount targets (free); what is metered is sandbox session time and the S3 storage the workspace uses. Off by default because it still adds resources you should not create by accident.
- **Cognito, DynamoDB (on-demand)** — the identity/state plane; negligible at demo volume.

`scripts/destroy.sh` removes everything `deploy.sh` created, including the OAuth credential provider and both gateways. Costs are usage-driven; an idle deployment still accrues the two microVMs' memory-time and the two static per-secret charges.

## Extending the agent

The four extension points that need no new plumbing:

| To add | Do this |
|---|---|
| A tool server | Create `mcp-servers/<name>/` with a `Dockerfile`, a server speaking MCP on `:8000`, and a `runtime.env` (`RUNTIME_SUFFIX`, `IMAGE_TAG`, `RUNTIME_ENV_MAP`, optional `DEPLOY_IF` gate and `REQUIRE_VARS`), then `./deploy.sh mcp <name>`. `scripts/build-mcp.sh` needs no edit — each server describes itself. The agent must be pointed at the new Runtime's URL, the way `APPROVAL_MCP_URL` is in `scripts/provision.sh` |
| A downstream system with its own login | Register an OAuth credential provider, add an MCP server, and one entry each in `IDP_REGISTRY` (router) and `_DOWNSTREAMS` (agent) — `/auth` then reports it, `/auth <key>` consents to it, and its tools ask for that system's consent. Built-in vendors need no shim; anything non-standard needs one like `lambda/shim/`. Recipe: `docs/per-user-oauth.md` |
| A different model | `MODEL_ID` in `.env`, then `./deploy.sh runtime` |
| A different system prompt | `AGENT_SYSTEM_PROMPT` in `.env`, passed through to the Runtime by `./deploy.sh runtime`. Blank keeps the default in `agent/agent_core.py`, which is short on purpose |

Two structural constraints shape anything larger. **One Runtime per MCP server** — `protocolConfiguration.serverProtocol` is a single value and a container exposes one MCP endpoint, so tools are grouped by trust boundary rather than packed together, which is why `lark-cli` (acts as the user) and `approval` (acts as the app) are separate servers. And **a Runtime-hosted MCP server cannot receive a per-user token from the Gateway**, so the agent fetches tokens itself and passes them in a custom header; a tool server on an addressable HTTPS endpoint could use the managed Gateway path instead. Both are explained in `docs/agentcore-behavior.md`.
