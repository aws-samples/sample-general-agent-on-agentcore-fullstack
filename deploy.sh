#!/usr/bin/env bash
# Deploy this sample end to end.
#
#   ./deploy.sh              everything, in dependency order (the normal case)
#   ./deploy.sh <step>       one step: base | mcp | 3lo | gateway | runtime | lark
#                            | approvals (subscribe to Lark approval events)
#   ./deploy.sh mcp [name...] only those MCP servers (default: all of mcp-servers/)
#   ./deploy.sh urls         reprint what to register in Lark, and the web page's sign-in
#   ./deploy.sh preflight    check tools, credentials and .env without deploying
#
# Target comes from .env (PROFILE / REGION / MODEL_ID); command-line env vars win:
#   REGION=ap-northeast-1 ./deploy.sh
#   FILES_STORAGE=true ./deploy.sh    code execution + per-user file mounts (creates a VPC, no NAT)
#   NAT_GATEWAY=true ./deploy.sh      with FILES_STORAGE, add a NAT so the code sandbox has internet
#   AGENT_BUILD=local ./deploy.sh     build the agent image with this host's Docker, not CodeBuild
#
# Nothing here waits for human input: the two values you have to paste into the
# Lark console don't block the deploy, they just gate the bot at runtime, so they
# are collected and printed at the end. Implementations live in scripts/ — this
# file owns the order, and every step is idempotent, so re-running is safe.
set -euo pipefail
# Parsed whole before any of it runs: bash otherwise reads a script as it goes, so an edit
# made while this runs (an editor, another session) shifts what it reads next — measured as
# a syntax error halfway through a deploy.
{
. "$(cd "$(dirname "$0")" && pwd)/scripts/lib/env.sh"

step() { printf '\n\033[1;36m### %s\033[0m\n' "$*"; }
note() { printf '\033[1;33m%s\033[0m\n' "$*"; }

run_preflight() { step "preflight";                       scripts/preflight.sh; }
run_base()    { step "base — CDK stacks";                scripts/provision.sh --base; }
run_mcp()     { step "mcp — MCP servers";                 scripts/build-mcp.sh "$@"; }
run_3lo()     { step "3lo — Identity + OAuth provider";  scripts/setup-3lo.sh; }
run_gateway() { step "gateway — Web Search (optional)";  scripts/provision.sh --gateway; }
run_code()    { step "code — Code Interpreter for generated code"; scripts/provision.sh --code; }
run_a2a()     { step "a2a — expose to peer agents";       scripts/provision.sh --a2a; }
run_webui()   { step "webui — static web chat";            scripts/provision.sh --webui; }
run_runtime() { step "runtime — agent (and A2A)";        scripts/provision.sh --runtime; }
run_lark()    { step "lark — credentials + allowlist";   scripts/setup-lark.sh; }
# No-ops unless AGENT_DECIDE_APPROVAL_CODES is set, so it costs nothing in the common
# case — and without it Lark silently never delivers the events.
run_approvals() { step "approvals — event subscription";  scripts/subscribe-approvals.sh; }

# What the user still has to do by hand, gathered from deployed state rather than
# scraped from the logs above.
print_urls() {
  local webhook callback
  webhook="$(aws cloudformation describe-stacks --stack-name "$PREFIX-router" \
    --query "Stacks[0].Outputs[?OutputKey=='WebhookLarkUrl'].OutputValue" \
    --output text 2>/dev/null || true)"
  callback="$(aws bedrock-agentcore-control get-oauth2-credential-provider \
    --name "$PREFIX-3lo" --query callbackUrl --output text 2>/dev/null || true)"

  printf '\n\033[1;32m=== Register these in the Lark developer console ===\033[0m\n'
  note "Events & Callbacks → Request URL:"
  echo "  ${webhook:-<not deployed yet>}"
  note "Security Settings → Redirect URLs:"
  echo "  ${callback:-<not deployed yet>}"
  echo
  echo "Then publish a version. See README.md → 'Lark console setup' for the"
  echo "permission scopes and the im.message.receive_v1 event subscription."
}

# Where to sign in, and as whom, when the web page is deployed.
print_web() {
  local url admin
  url="$(aws cloudformation describe-stacks --stack-name "$PREFIX-webui" \
    --query "Stacks[0].Outputs[?OutputKey=='WebUiUrl'].OutputValue" --output text 2>/dev/null || true)"
  [ -n "$url" ] && [ "$url" != "None" ] || return 0
  admin="$(aws secretsmanager get-secret-value --secret-id "$PREFIX/web/admin" --query SecretString \
    --output text 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["username"], d["password"])' 2>/dev/null || true)"
  printf '\n\033[1;32m=== Web chat ===\033[0m\n'
  echo "  $url"
  if [ -n "$admin" ]; then
    echo "  username: ${admin% *}"
    echo "  password: ${admin#* }"
  fi
  echo "  invite others on the page's 管理 view (admin.html)"
}

usage() { sed -n '2,16p' "$0" | cut -c3-; }

case "${1:-all}" in
  base)    run_base ;;
  mcp)     shift; run_mcp "$@" ;;
  3lo)     run_3lo ;;
  gateway) run_gateway ;;
  runtime) run_runtime ;;
  lark)    run_lark ;;
  approvals) run_approvals ;;
  code)    run_code ;;
  a2a)     run_a2a ;;
  webui)   run_webui ;;
  urls)    if [ "$LARK" = "true" ]; then print_urls; fi; print_web ;;
  preflight) run_preflight ;;
  all|"")
    run_preflight
    # 3lo and gateway precede runtime: the agent is deployed with the provider,
    # workload and gateway URL baked into its environment.
    run_base; run_mcp; run_3lo; run_gateway; run_code; run_runtime
    # Without a Lark app the web page is the whole surface: nothing to register in Lark.
    if [ "$LARK" = "true" ]; then run_lark; run_approvals; fi
    run_webui
    if [ "$LARK" = "true" ]; then print_urls; fi
    print_web
    ;;
  -h|--help) usage ;;
  *) echo "unknown step: $1"; echo; usage; exit 1 ;;
esac
exit
}
