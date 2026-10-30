# 5. 常见问题

## 预检报错

`./deploy.sh preflight` 和 `./deploy.sh` 开头的预检会在出错时停下，并打印下面某一条：

| 提示 | 原因与处理 |
|---|---|
| `missing tools: …` | 列出的工具没装或不在 `PATH` 里，提示里附了安装地址。`npx` 来自 Node.js |
| `tools too old: …` | 列出的工具版本太低，按提示升级，版本要求见 [1.1](01-prepare.md#11-本机工具与-aws-凭证) |
| `AWS CLI … has no command for: …` | AWS CLI 太旧，还不认识 AgentCore 等新服务，升级到最新的 AWS CLI v2 |
| `uv could not set up the Python environment…` 或 `could not fetch the CDK CLI from npm…` | 下载失败，见下面的[预检下载很慢或卡住](#预检下载很慢或卡住) |
| `AWS credentials not usable…` | 环境变量里的临时凭证过期了（重新从 Workshop Studio 复制），或者 `.env` 里的 `PROFILE` 不存在。用 `aws sts get-caller-identity` 确认凭证可用 |
| `Lark is half configured — unset in .env: LARK_ADMIN_OPEN_ID` | `.env` 里 Lark 的示例值没清空。把 `LARK_APP_ID`、`LARK_APP_SECRET`、`LARK_ENCRYPT_KEY`、`LARK_VERIFICATION_TOKEN` 全部清空，见 [1.3](01-prepare.md#13-填写-env) |
| `no Lark app and WEBUI is not true — nobody could reach the agent` | 不接 Lark 时 Web 页面是唯一入口，把 `WEBUI` 设为 `true` |
| `GitHub is half configured — set both GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET in .env, or neither` | 两项要同时填写 |
| `DEPLOYMENT_NAME '…' must be 3-23 chars of a-z, 0-9 and '-', starting with a letter` | 按规则改名 |

还有三条只是警告，预检不会停下：

- `warning: Node.js … is not one the CDK supports (22, 24)`：其他版本一般也能部署，但 CDK 会打印一大段警告；部署出现异常时换成 22 或 24 LTS。

- `warning: can't confirm <MODEL_ID> in <REGION> — enable model access if the agent 403s`：在 Bedrock 控制台为该区域开通模型访问。不开通的话部署能完成，但 Agent 回答时会报 403。默认的 Kimi K3 是第三方模型，第一次调用可能要在 AWS Marketplace 订阅；账号不允许订阅时，在 `.env` 里把 `MODEL_ID` 换成账号可用的其他模型。
- `warning: no read access to: …`：当前凭证读不了列出的服务。部署要写这些服务，大概率会失败，先换一个权限足够的凭证。

## 预检下载很慢或卡住

第一次预检要下载 Python 和约 300 MB 的依赖（`python  : first run — downloading…` 之后的进度），然后从 npm 取 CDK CLI（`cdk     :` 那一行）。网络慢或访问 PyPI、GitHub、npm 受限时，按 Ctrl-C 停下，在同一个终端设置镜像后重新运行 `./deploy.sh preflight`：

```bash
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
export UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone/
export npm_config_registry=https://registry.npmmirror.com
```

用了 PyPI 镜像后 `uv.lock` 会被改写成镜像地址，不影响部署。

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

## 访问限制相关

- **页面右上角没有「访问控制」**：只有 `admins` 组的用户能看到它。`admin@agentcore.local` 在这个组里；你邀请的用户默认不在。
- **状态行显示「拦截规则与列表不一致」或「拦截规则未生效」**：结果过滤已经按列表生效，只是 Policy 里的规则没跟上，比如上一次修改还在校验时又改了一次。点击 **重新应用**。
- **添加或移除时提示「the previous change is still being applied」**：上一次修改的规则还在校验，过几秒再试一次。
- **Agent 说没收到查询词，也没有调用工具**：模型偶尔会这样。发送 `/new` 后再发一次。
- **D3 的搜索没有被拒**：先确认状态行已经显示「已生效」。规则只在域名能出现的位置匹配，且区分大小写：`ietf.org`、`datatracker.ietf.org` 会被拒，`IETF.ORG` 或紧贴在中文后面的写法不会；这些情况下结果仍会被过滤。

## 其它

- **忘记管理员密码**：运行 `./deploy.sh urls`，重新打印页面地址和管理员账号。
- **第一条消息回答得慢**：新会话的第一个请求要启动 microVM、连接工具，比之后的请求慢。
