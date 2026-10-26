# 5. 常见问题

## 预检报错

`./deploy.sh preflight` 和 `./deploy.sh` 开头的预检会在出错时停下，并打印下面某一条：

| 提示 | 原因与处理 |
|---|---|
| `missing tools: …` | 列出的工具没装或不在 `PATH` 里。`npx` 来自 Node.js |
| `AWS credentials not usable…` | 环境变量里的临时凭证过期了（重新从 Workshop Studio 复制），或者 `.env` 里的 `PROFILE` 不存在。用 `aws sts get-caller-identity` 确认凭证可用 |
| `Lark is half configured — unset in .env: LARK_ADMIN_OPEN_ID` | `.env` 里 Lark 的示例值没清空。把 `LARK_APP_ID`、`LARK_APP_SECRET`、`LARK_ENCRYPT_KEY`、`LARK_VERIFICATION_TOKEN` 全部清空，见 [1.3](01-prepare.md#13-填写-env) |
| `no Lark app and WEBUI is not true — nobody could reach the agent` | 不接 Lark 时 Web 页面是唯一入口，把 `WEBUI` 设为 `true` |
| `GitHub is half configured — set both GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET in .env, or neither` | 两项要同时填写 |
| `DEPLOYMENT_NAME '…' must be 3-23 chars of a-z, 0-9 and '-', starting with a letter` | 按规则改名 |

还有两条只是警告，预检不会停下：

- `warning: can't confirm <MODEL_ID> in <REGION> — enable model access if the agent 403s`：在 Bedrock 控制台为该区域开通模型访问。不开通的话部署能完成，但 Agent 回答时会报 403。默认的 Kimi K3 是第三方模型，第一次调用可能要在 AWS Marketplace 订阅；账号不允许订阅时，在 `.env` 里把 `MODEL_ID` 换成账号可用的其他模型。
- `warning: no read access to: …`：当前凭证读不了列出的服务。部署要写这些服务，大概率会失败，先换一个权限足够的凭证。

## 部署失败

每个步骤都是幂等的，修复问题后重新运行 `./deploy.sh` 即可，已完成的部分会被跳过或原样更新。也可以只重跑失败的那一步，例如 `./deploy.sh runtime`。

和别人共用一个 AWS 账号时，每个人的 `DEPLOYMENT_NAME` 必须不同，否则会操作到同一套资源。每套部署要一个 VPC 和一个 NAT gateway（占一个弹性 IP），而一个区域默认最多 5 个 VPC、5 个弹性 IP，默认 VPC 已占一个，所以一个账号最多部署 4 套；人更多时先提高这两项配额。

## GitHub 报 redirect_uri_mismatch

点击「授权 GitHub」后，GitHub 页面显示 `redirect_uri_mismatch`（The redirect_uri MUST match the registered callback URL for this application）：OAuth App 里的 Redirect URI 和 AgentCore Identity 的回调地址不一致。

1. 用 [3.1](03-callback-and-login.md#31-查询-github-回调地址) 的命令重新查询回调地址。
2. 粘贴到 OAuth App 的 Redirect URI，点击 Update application 保存。注意不要多出空格或换行。
3. 回到聊天页面，再点一次「授权 GitHub」。

清理后重新部署时，provider 是重新创建的，回调地址会变，需要重新填写。

## 看不到组织的仓库

如果某个组织开启了 OAuth App 访问限制，在组织所有者批准你的 OAuth App 之前，Agent 看不到这个组织的仓库，只能看到你的个人仓库。本实验用个人仓库完成即可。

## /auth 里没有 GitHub，或者回复「GitHub 未配置」

部署时 `.env` 里还没有 GitHub 的两项，或者部署之后才填。填好后依次运行：

```bash
./deploy.sh 3lo
./deploy.sh runtime
```

`3lo` 创建 GitHub 的 OAuth provider，并加入 `/auth` 的列表；`runtime` 让 Agent 带着这个 provider 重新部署。完成后按 [3.1](03-callback-and-login.md#31-查询-github-回调地址) 查询回调地址，填回 GitHub。

## 授权相关

- **点击「授权 GitHub」后没有打开新标签页**：浏览器拦截了弹窗时，页面会在当前标签页打开 GitHub 授权页，授权完成后回到聊天页面。
- **提示「授权没有完成：…，请再点一次授权」**：这次授权的会话已失效，再点一次按钮即可。
- **想用另一个 GitHub 账号授权**：先发送 `/auth revoke github`，再到 GitHub 的 **Settings → Applications → Authorized OAuth Apps** 撤销这个 App，然后在浏览器里登录另一个 GitHub 账号再授权。
- **在 GitHub 那边撤销授权后，回答一直停在「调用工具」**：先刷新页面，再发送 `/new`，然后重新提问，会出现「授权 GitHub」按钮。要在 GitHub 那边撤销授权，先发送 `/auth revoke github` 再去撤销，就不会遇到这个问题。

## 其它

- **忘记管理员密码**：运行 `./deploy.sh urls`，重新打印页面地址和管理员账号。
- **第一条消息回答得慢**：新会话的第一个请求要启动 microVM、连接工具，比之后的请求慢。
