# Workshop：在 Amazon Bedrock AgentCore 上构建企业级通用 Agent

在自己的 AWS 账号里，用一条命令部署一个运行在 Amazon Bedrock AgentCore 上的通用 Agent。它有带登录的 Web 聊天入口，能联网搜索、运行代码、记住你，还能以你的身份访问 GitHub。然后逐项验证这些能力背后用到的 AgentCore 组件。

## 实验目标

- **部署**：一条 `./deploy.sh` 完成全部部署，包括 Agent Runtime、Web 聊天页面与 Cognito 登录、AgentCore Memory、Code Interpreter 沙箱、Web 搜索网关和 AgentCore Identity。
- **入口与身份**：用户在 Cognito 托管登录页登录，Agent 校验的是用户自己的 token；管理员在页面上管理谁能登录。
- **会话与对话**：理解 AgentCore Runtime 的会话和 microVM，以及对话记录为什么能在换实例后保留；回答过程中可以继续补充消息。
- **工具**：Web 搜索（AgentCore Gateway）、代码执行和每个用户的持久工作区（Code Interpreter + S3 Files）、长期记忆（AgentCore Memory）。
- **以用户身份调用下游系统**：工具第一次需要 GitHub 时，在聊天里完成 OAuth 授权，token 存进 AgentCore Identity 的 Token Vault；之后 Agent 带着你自己的 token 调用 GitHub。
- **清理**：实验结束后完整删除资源。

本 workshop 不接 Lark 和 Google：它们的 OAuth 应用需要管理员在后台配置。下游系统只用 GitHub，几乎每个开发者都有账号，创建 OAuth App 也不需要审批。Lark 和 Google 的接入方式见 [docs/console-setup.md](https://github.com/aws-samples/sample-general-agent-on-agentcore-fullstack/blob/main/docs/console-setup.md)。

## 时间安排

Workshop 先有 30 分钟方案介绍，然后是 60 分钟动手实验。60 分钟内完成部署和必做的验证任务；选做任务和清理可以会后继续：Workshop Studio 提供的实验环境保留 2 天。

| 时间 | 内容 | 章节 |
|---|---|---|
| 0–10 分钟 | 安装工具、配置凭证，创建 GitHub OAuth App，填写 `.env`，运行预检 | [1. 准备](01-prepare.md) |
| 10–45 分钟 | 运行 `./deploy.sh`（约 35 分钟）。等待时阅读[部署架构](00-architecture.md)，了解正在部署的每个资源 | [2. 部署](02-deploy.md) |
| 45–50 分钟 | 把回调地址填回 GitHub，登录 Web 聊天页面 | [3. 回调地址与登录](03-callback-and-login.md) |
| 50–60 分钟 | 开始验证任务，没做完的会后继续 | [4. 验证任务](04-tasks.md) |
| 会后 | 选做任务；用自己的 AWS 账号时清理资源 | [4. 验证任务](04-tasks.md)、[6. 清理](06-cleanup.md) |

遇到问题随时查 [5. 常见问题](05-troubleshooting.md)。

## 前置条件

- **AWS 账号**：有权限创建 IAM 角色、CloudFormation 栈、VPC、Lambda、Cognito、DynamoDB、S3、CloudFront、CodeBuild、ECR 和 Bedrock AgentCore 资源，并且能在 `us-east-1` 创建 AgentCore Gateway（Web 搜索连接器只在该区域提供）。workshop 提供的账号已满足这些要求。
- **Bedrock 模型访问**：在部署区域 `us-west-2` 能调用 `.env` 里 `MODEL_ID` 指定的模型（默认 `global.moonshotai.kimi-k3`）。
- **GitHub 账号**：个人账号即可，不需要 Copilot 订阅。
- **本机工具**：`git`、AWS CLI v2、Node.js 22 或 24 LTS（提供 `npx`，用来运行 CDK）、Python 3、[uv](https://docs.astral.sh/uv/)。不需要 Docker：镜像在 CodeBuild 里构建。
- **操作系统**：macOS 或 Linux。部署脚本是 bash，Windows 需要在 WSL2 里操作。

## 你会部署什么

```text
浏览器 ── Web 聊天页面（S3 + CloudFront），Cognito 托管登录
   │  用户自己的 Cognito token
   ▼
Agent Runtime（AgentCore Runtime，每个会话一个 microVM）
   ├─ 对话记录 ──────── DynamoDB + S3 checkpoint
   ├─ 长期记忆 ──────── AgentCore Memory
   ├─ 代码执行 ──────── AgentCore Code Interpreter + S3 Files 工作区（经 NAT 联网）
   ├─ Web 搜索 ──────── AgentCore Gateway（us-east-1）
   └─ GitHub 工具 ───── GitHub 官方 MCP server，带着从 AgentCore Identity 取出的你的 token
```

具体部署哪些资源、三条请求路径、数据存放位置和费用，见[部署架构](00-architecture.md)；更深入的设计见 [docs/architecture.md](https://github.com/aws-samples/sample-general-agent-on-agentcore-fullstack/blob/main/docs/architecture.md)。
