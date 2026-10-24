[English](README.md) | [简体中文](README.zh-CN.md)

# General Agent on AgentCore — 全栈 Agent 示例

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

一个基于 **Amazon Bedrock AgentCore** 的通用 Agent,可通过 **Lark(飞书)聊天**、网页、以及 A2A 对等 Agent 三种渠道访问。

本示例要证明的核心特性只有一条:**每个工具都以提问者本人的身份行事。** 一条消息会解析为唯一的 Lark 用户,任何触及其数据的工具调用下游服务(Lark、Google)时,使用的都是*该用户自己的 OAuth token*,由 AgentCore Identity 托管。权限判定交给下游服务本身,因此不存在需要额外同步的第二套权限系统。

> **免责声明。** 本代码为示例代码,不用于生产环境。部署前请与你的安全和法务团队确认是否满足组织的安全、合规要求。防护范围与边界见 [docs/security.md](docs/security.md)。

![架构图——端到端的按用户身份隔离](docs/images/architecture.png)

<details>
<summary>文字版架构图——便于在终端或由 AI 阅读</summary>

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

## 我们构建了什么

十个层次,每一层都是真实实现而非占位——这正是名字里"full-stack"的含义。真正有意思的是层与层之间的接缝。

| 层次 | 它决定什么 | 如何实现 |
|---|---|---|
| **渠道(Channel)** | 一轮对话如何开始、答案如何返回 | Lark webhook → router Lambda → 答案以流式方式写入 CardKit 卡片。另外两个渠道接到同一个 Agent:Lark 内嵌页面(AG-UI over SSE)、面向对等 Agent 的 A2A |
| **模型(Model)** | 由哪个模型回答 | **Bedrock** —— 走 Converse API;默认是开放权重模型 Kimi K3,`MODEL_ID` 可换成任意支持流式工具调用的模型;模型支持时启用提示缓存(`cachePoint`) |
| **Agent 循环** | 提示词,以及工具调用如何迭代 | LangGraph `create_agent`,配合按阈值触发的摘要压缩,为长期对话线程设置上限 |
| **身份(Identity)** | 这一轮是谁、它的工具可以用谁的凭证 | Cognito 签发身份;**AgentCore Identity** 按下游系统分别托管每个人的授权;工具以用户本人身份行事 |
| **会话状态** | 一次对话*内部*记住什么 | DynamoDB checkpoint,每个 superstep 都写入,因此在 15 分钟上限处被截断的一轮对话可以续接 |
| **长期记忆** | *跨*对话记住什么 | **AgentCore Memory**,通过 `remember`/`recall`——只有当模型判断某个事实值得跨对话保留时才写入 |
| **持久化存储** | 什么内容在容器之外仍然存在、归谁所有 | **S3 Files**,每个用户一个 Access Point,挂载到 `/mnt/workspace`;隔离边界是 microVM,而不是 Agent 代码 |
| **工具(Tools)** | 它能做什么、以谁的身份做 | 各自独立 Runtime 上的 **MCP server**——`lark-cli` 和 `google` 以调用者本人身份行事,审批相关操作用应用身份;**Gateway** 提供网络搜索;**Code Interpreter** 运行生成的代码,并用 `deliver_file` 把产物发进聊天、用 `save_attachment` 打开用户发来的文件 |
| **可观测性** | 它做了什么、是否成功 | **AgentCore Observability** 提供 trace 和日志;一个 CloudWatch dashboard 叠加平台指标与 Agent 自身指标——对话结果、按名称统计的工具调用、自我修复次数 |
| **运维(Operations)** | 如何部署与销毁 | CloudFormation 覆盖的部分用 CDK,其余用控制面 API 调用,两者都由同一个 `./deploy.sh` 按顺序编排 |

有意不使用两个 AgentCore 原语:Browser(这里不需要无头浏览器)和 Evaluations——回答质量是一个独立课题,有自己的项目 [cs-agent-eval-loop](https://github.com/aleck31/cs-agent-eval-loop),本仓库只衡量"机制本身是否跑通"。

四个层默认关闭,因为各自会带来额外成本或控制台配置工作:`WEB_SEARCH`、`WEBUI`、`A2A`、以及带存储的代码执行(`FILES_STORAGE`)。不开启任何一个,Agent 依然能正常运行。

## 目录结构

| 路径 | 内容 |
|---|---|
| `app.py`、`cdk.json` | CDK 应用(依赖由 uv 管理)。部署状态写入 `.cdk-state.<DEPLOYMENT_NAME>.json`,不在这里 |
| `.env` | 部署目标(`PROFILE`/`REGION`/`MODEL_ID`)+ Lark 凭证——已加入 .gitignore,每个脚本都会读取 |
| `stacks/` | security、agentcore、router、shim、gateway、observability,以及开启对应 flag 时的 storage 和 webui |
| `agent/` | Agent 容器:HTTP 契约 + DynamoDB checkpoint + 按用户的 3LO(`lark_3lo`)+ 面向 lark-cli server 的 MCP 客户端,以及可选的网络搜索(`websearch`);在后台运行每一轮对话,并将答案流式写入卡片(`lark_notify`)。`code_tools.py` 在按用户隔离的沙箱中运行生成的代码;`agui.py` 在同一个 Runtime 上为浏览器提供服务;`a2a_server.py` 是同一镜像上的第二个入口(`SERVER_MODE=a2a`) |
| `webui/` | 网页聊天页面——单个静态文件,无需构建步骤;部署时的配置值写入 `config.js` |
| `lambda/router/` | Lark webhook:校验/解密/租户 token/发送 + 3LO 授权等待 + 聊天命令。也是唯一签发按用户 JWT 的组件(`cognito.py`) |
| `lambda/shim/` | Lark OAuth 的 RFC-6749 适配层 + 3LO 回调端点(`CompleteResourceTokenAuth`,随后给用户发私信) |
| `mcp-servers/` | 每个 MCP server 一个目录,各自一个 Runtime:`lark-cli/` 以用户身份操作 Lark,`google/` 对 Google 做同样的事(第二个下游系统,用于验证身份链并非 Lark 专属),`approval/` 以应用身份执行审批决策。每个 server 在 `runtime.env` 中自描述构建/运行配置,因此新增一个 server 不需要改脚本 |
| `deploy.sh` | 部署入口——按顺序编排 `scripts/` 里的各个步骤 |
| `scripts/` | 各步骤的具体实现:preflight / provision(base/memory/code/runtime/gateway/a2a/webui)/ build-mcp / setup-3lo / setup-lark / subscribe-approvals / manage-allowlist / destroy |
| `tests/` | `run.sh`(全部单元测试套件)+ 需要已部署环境的端到端 smoke test |
| `mcp-servers/probe/` | 一个仅用于测量的 MCP server,不是产品部署的一部分——`docs/agentcore-behavior.md` 里的结论就是靠它建立的 |

## 快速开始

需要 `uv`、AgentCore CLI(`npm i -g @aws/agentcore`),以及 AWS 凭证。镜像在 CodeBuild 里以 ARM64 原生构建;只有设置 `AGENT_BUILD=local` 时才需要 Docker。

```bash
cp .env.example .env    # 部署目标 + Lark 应用凭证 + 你自己的 open_id
./deploy.sh             # 按依赖顺序部署全部内容
tests/run.sh            # 单元测试,不需要 AWS
```

`./deploy.sh` 不会中途停下来询问你任何问题。结束时会打印两个 URL,需要粘贴到 Lark 控制台——它们不会阻塞部署,只是在运行时为 bot 把关(`./deploy.sh urls` 可以重新打印)。

首次运行前有三件事需要了解:

- **Lark 应用的 scope、事件订阅、回调地址需要手动配置** → [docs/console-setup.md](docs/console-setup.md)。Google 是可选项,配置说明在同一篇文档里。
- **每个步骤都是幂等的**(`./deploy.sh base|mcp|3lo|runtime|lark|…`),顺序只在一处有讲究 → [docs/deploy.md](docs/deploy.md)。
- **用 `scripts/destroy.sh` 销毁环境。** 删除 OAuth provider 会清空每个用户已授权的 token,并生成一个新的回调 URL 需要重新登记。

然后给 bot 发消息。`/help` 会列出聊天内可用命令:

| 命令 | 作用 |
|---|---|
| `/auth`、`/auth <idp>` | 查看各下游系统的授权状态;授权或重新授权某一个 |
| `/status` | 你的身份、两个会话 id、当前服务你的 microVM、最近活跃时间,以及其余各项该用哪个命令查看 |
| `/context` | 对话轮数与工具调用次数、最近一次请求实际计费的上下文大小、系统提示词、上下文压缩设置、prompt cache 命中率 |
| `/model` | 当前服务你的容器所用的模型、region 与 prompt cache 方式 |
| `/tools` | 这段对话实际拥有的工具,按来源分组;未授权的系统会提示要发送哪个 `/auth` |
| `/compact` | 立即把较早的对话压缩成摘要,保留最近几条消息 |
| `/new`、`/reset`、`/reconnect` | 同时轮换两个 id / 只轮换对话线程 / 只轮换运行时实例 |
| `/clear` | 真正删除当前线程的 checkpoint |

运行时会话(runtime session)与 checkpoint 线程是**两个独立的 id**——一个决定由哪个 microVM 为你服务,另一个决定 Agent 读取哪段历史。授权则是第三个、互相独立的维度,按用户绑定,所以 `/new` 从不需要重新授权。

## 成本

所有常驻资源都是按量计费的,单用户空闲演示环境月成本大约是几美元(不含模型调用费用);Bedrock 调用是可变成本中的主要部分。按服务拆分的明细、以及空闲时仍会产生的两笔费用:[docs/deploy.md](docs/deploy.md#cost)。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 完整流程、每一跳的鉴权、授权等待的时序、十个层次、仓库结构 |
| [docs/deploy.md](docs/deploy.md) | 每个部署步骤、可选功能、销毁方式、成本、扩展点 |
| [docs/console-setup.md](docs/console-setup.md) | Lark 与 Google 控制台配置 |
| [docs/security.md](docs/security.md) | 安全态势与已知边界 |
| [docs/agentcore-behavior.md](docs/agentcore-behavior.md) | 实测得到的 AgentCore 行为——**排查任何平台层问题前请先读这篇** |
| [docs/per-user-oauth.md](docs/per-user-oauth.md) | 工具如何以用户身份访问另一个应用,以及如何新增一个下游系统 |

身份链本身的详细说明在另外两个姊妹示例里,而不是本仓库:[native](https://github.com/aws-samples/sample-lark-identity-on-agentcore-native)(本仓库采用的 Token Vault 路径)和 [interceptor](https://github.com/aws-samples/sample-lark-identity-on-agentcore-interceptor)(同样的保证,自行管理 token 托管)。
