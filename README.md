[English](README.md) | [简体中文](README.zh-CN.md)

# General Agent on AgentCore — a full-stack agent sample

<p align="center">
  <img alt="Model platform" src="https://img.shields.io/badge/Model%20platform-Amazon%20Bedrock-ff9900?style=flat-square&logo=amazonaws&logoColor=white">
  <img alt="Agent framework" src="https://img.shields.io/badge/Agent%20framework-LangGraph-1c3c3c?style=flat-square">
  <img alt="Runtime" src="https://img.shields.io/badge/Runtime-AgentCore%20V2-ff9900?style=flat-square&logo=amazonaws&logoColor=white">
  <img alt="Identity" src="https://img.shields.io/badge/Identity-AgentCore%20Token%20Vault-ff9900?style=flat-square&logo=amazonaws&logoColor=white">
  <img alt="Channel" src="https://img.shields.io/badge/Channel-Lark%20%C2%B7%20Web%20%C2%B7%20A2A-00d6b9?style=flat-square">
  <img alt="Tools" src="https://img.shields.io/badge/Tools-MCP-6f42c1?style=flat-square">
  <img alt="IaC" src="https://img.shields.io/badge/IaC-CDK%20%28Python%29-ff9900?style=flat-square&logo=amazonaws&logoColor=white">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.13-3776ab?style=flat-square&logo=python&logoColor=white">
  <img alt="Node" src="https://img.shields.io/badge/Node-24-339933?style=flat-square&logo=nodedotjs&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT--0-3da639?style=flat-square">
</p>

A general-purpose agent on **Amazon Bedrock AgentCore**, reachable from **Lark (Feishu) chat**, a web page, and peer agents over A2A.

The point of the sample is one property: **every tool acts as the person who asked.** A message resolves to one Lark user, and any tool touching their data calls the downstream service — Lark, Google — with *that user's own OAuth token*, vaulted by AgentCore Identity. The service adjudicates access, so there is no second permission system to keep in sync.

> **Disclaimer.** This is sample code, for non-production usage. You should work with your security and legal teams to meet your organizational security, regulatory and compliance requirements before deployment. See [docs/security.md](docs/security.md) for what it does and does not defend against.

![Architecture — per-user identity, end to end](docs/images/architecture.png)

<details>
<summary>Text version — readable in a terminal or by an agent</summary>

```
                                        ┌──────────── AgentCore Identity ────────────┐
                                        │  Token Vault: stores / refreshes / returns │
                                        │  THIS user's Lark token (3LO)              │
                                        └──────────────────┬─────────────────────────┘
  Lark bot chat ──webhook──▶ Router Lambda                 │ the user's own token
  Lark h5 page  ──h5 code──▶ (mints the JWT)               │
  Peer agent    ──bearer───▶ A2A Runtime                   ▼
                                  └──────▶  Agent (AgentCore Runtime)  ──▶  Lark MCP server
                                            LangGraph + Bedrock             acts AS the user
                                            DynamoDB checkpoints                    │
                                              │          │                          ▼
   the answer, when the turn finishes  ◀──────┘          │                   Lark REST API
   (chat card · SSE · A2A reply)                         │            returns only what THIS
                                                         ▼            user can see
                                          Code Interpreter session
                                          /mnt/workspace ← this user's S3 Files
```

</details>

## What we build

Ten layers, each with a real implementation rather than a stub — that is the "full-stack" in the name. The interesting part is the seams between them.

| Layer | What it decides | How it's built |
|---|---|---|
| **Channel** | how a turn starts, how the answer returns | Lark webhook → router Lambda → answer typed into a streaming CardKit card. Two more surfaces reach the same agent: a page inside Lark (AG-UI over SSE) and A2A for peers |
| **Model** | which model answers | **Bedrock** — Claude Sonnet over Converse, with an explicit `cachePoint` on every request |
| **Agent loop** | the prompt, and how tool calls iterate | LangGraph `create_agent`, plus threshold-triggered summarisation to bound a permanent thread |
| **Identity** | who this turn is, and whose credentials its tools may use | Cognito mints the signed identity; **AgentCore Identity** vaults each person's grant per downstream system; tools act as them |
| **Session state** | what it remembers *within* a conversation | DynamoDB checkpoints, written per superstep, so a turn cut off at the 15-minute cap resumes |
| **Long-term memory** | what it remembers *across* conversations | **AgentCore Memory** via `remember`/`recall` — written only when the model decides a fact should outlast the chat |
| **Persistent storage** | what survives the container, and whose it is | **S3 Files**, one Access Point per user, mounted at `/mnt/workspace`; isolated at the microVM boundary, not by agent code |
| **Tools** | what it can do, and as whom | **MCP servers** on their own Runtimes — `lark-cli` and `google` as the calling user, approvals on the app identity; **Gateway** for web search; **Code Interpreter** for generated code |
| **Observability** | what it did, and whether it worked | **AgentCore Observability** for traces and logs; a CloudWatch dashboard over platform metrics plus the agent's own — turn outcomes, tool calls by name, self-repairs |
| **Operations** | how it deploys and tears down | CDK for what CloudFormation covers, control-plane calls for the rest, both ordered by one `./deploy.sh` |

Two AgentCore primitives are deliberately **not** used: Browser (nothing here needs a headless one) and Evaluations — answer quality is a separate concern with its own project, [cs-agent-eval-loop](https://github.com/aleck31/cs-agent-eval-loop), so this repo measures only whether the machinery ran.

Four layers ship switched off, because each adds cost or console work: `WEB_SEARCH`, `WEBUI`, `A2A`, and code execution with its storage (`FILES_STORAGE`). The agent runs without any of them.

## Layout

| Path | What |
|---|---|
| `app.py`, `cdk.json` | CDK app (uv-managed deps). Deployment state goes to `.cdk-state.json`, not here |
| `.env` | Deployment target (`PROFILE`/`REGION`/`MODEL_ID`) + Lark credentials — gitignored, read by every script |
| `stacks/` | security, agentcore, router, shim, gateway, observability, plus storage and webui when their flags are on |
| `agent/` | the agent container: HTTP contract + DynamoDB checkpoints + per-user 3LO (`lark_3lo`) + MCP clients for the lark-cli server and, optionally, web search (`websearch`); runs turns in the background and streams answers into a card (`lark_notify`). `code_tools.py` runs generated code in a per-user sandbox; `agui.py` serves the browser off the same Runtime; `a2a_server.py` is a second entrypoint on the same image (`SERVER_MODE=a2a`) |
| `webui/` | the web chat page — one static file, no build step; deploy-time values arrive as `config.js` |
| `lambda/router/` | Lark webhook: verify/decrypt/tenant-token/send + 3LO consent-wait + the chat commands. Also the only component that mints per-user JWTs (`cognito.py`) |
| `lambda/shim/` | Lark OAuth RFC-6749 façade + 3LO return endpoint (`CompleteResourceTokenAuth`, then DMs the user) |
| `mcp-servers/` | One directory per MCP server, one Runtime each: `lark-cli/` acts as the user against Lark, `google/` does the same against Google (the second system, added to test that the identity chain is not Lark-specific), `approval/` runs approval decisions on the app identity. Each declares its own build/runtime config in `runtime.env`, so adding a server needs no script change |
| `deploy.sh` | the deploy entry point — orders the steps in `scripts/` |
| `scripts/` | step implementations: preflight / provision (base/memory/code/runtime/gateway/a2a/webui) / build-mcp / setup-3lo / setup-lark / subscribe-approvals / manage-allowlist / destroy |
| `tests/` | `run.sh` (all unit suites) + e2e smoke tests that need a deployed stack |
| `probe/` | a measurement MCP server, not part of the deployed agent — how the findings in `docs/agentcore-behavior.md` were established |

## Quick start

Needs `uv`, Docker, the AgentCore CLI (`npm i -g @aws/agentcore`), and AWS credentials.

```bash
cp .env.example .env    # deployment target + Lark app credentials + your open_id
./deploy.sh             # everything, in dependency order
tests/run.sh            # unit suites, no AWS needed
```

`./deploy.sh` never stops to ask you anything. At the end it prints two URLs to paste into the Lark console — they don't block the deploy, they gate the bot at runtime (`./deploy.sh urls` reprints them).

Three things to know before the first run:

- **The Lark app needs scopes, events and redirect URLs set by hand** → [docs/console-setup.md](docs/console-setup.md). Google is optional and on the same page.
- **Individual steps are idempotent** (`./deploy.sh base|mcp|3lo|runtime|lark|…`), and the order matters in one place → [docs/deploy.md](docs/deploy.md).
- **Tear down with `scripts/destroy.sh`.** Deleting the OAuth provider purges every user's vaulted token and mints a new callback URL to register.

Then message the bot. `/help` lists the commands in chat:

| Command | What it does |
|---|---|
| `/auth`, `/auth <idp>` | Authorization status per downstream system; authorize or re-authorize one |
| `/status` | Identity, both session ids, the microVM serving you, the model it really calls, turns and tool calls |
| `/new`, `/reset`, `/reconnect` | Rotate both ids, the thread only, or the runtime instance only |
| `/clear` | Actually delete this thread's checkpoints |

The runtime session and the checkpoint thread are **independent ids** — one decides which microVM serves you, the other which history the agent reads. Authorization is a third, orthogonal dimension keyed to the user, so `/new` never costs a re-consent.

## Cost

Everything always-on is consumption-priced, so an idle single-user demo runs on the order of a couple USD/month before model usage; Bedrock calls dominate the variable cost. Per-service breakdown and the two charges that accrue while idle: [docs/deploy.md](docs/deploy.md#cost).

## Documentation

| Doc | What's in it |
|---|---|
| [docs/architecture.md](docs/architecture.md) | The full flow, per-hop auth, the consent-wait sequence, the ten layers, repo layout |
| [docs/deploy.md](docs/deploy.md) | Every deploy step, the optional features, teardown, cost, extension points |
| [docs/console-setup.md](docs/console-setup.md) | Lark and Google console configuration |
| [docs/security.md](docs/security.md) | Security posture and known limits |
| [docs/agentcore-behavior.md](docs/agentcore-behavior.md) | Measured AgentCore behaviour — **read this before debugging anything platform-level** |
| [docs/per-user-oauth.md](docs/per-user-oauth.md) | How a tool acts as the user in another app, and how to add one |

The identity chain itself is documented in depth by two sibling samples rather than here: [native](https://github.com/aws-samples/sample-lark-identity-on-agentcore-native) (the Token Vault path this repo uses) and [interceptor](https://github.com/aws-samples/sample-lark-identity-on-agentcore-interceptor) (same guarantees, self-managed vaulting).
