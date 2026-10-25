#!/usr/bin/env bash
# Check that a deploy can succeed, and print the target it would hit.
#
# Runs before the first slow step: a CodeBuild image takes minutes, so finding out
# afterwards that a Lark credential was blank means paying for that wait twice.
# Read-only — safe to run any time.
#
# Usage: [PROFILE=p REGION=r] scripts/preflight.sh
set -euo pipefail

. "$(cd "$(dirname "$0")" && pwd)/lib/env.sh"
WEB_SEARCH="${WEB_SEARCH:-false}"
AGENT_BUILD="${AGENT_BUILD:-codebuild}"

fail() { printf '\033[1;31m%s\033[0m\n' "$*" >&2; exit 1; }

# --- tools ---------------------------------------------------------------
missing=""
for t in uv agentcore aws npx python3 $([ "$AGENT_BUILD" = "local" ] && echo docker); do
  command -v "$t" >/dev/null || missing="$missing $t"
done
[ -z "$missing" ] || fail "missing tools:$missing
  uv: https://docs.astral.sh/uv/   agentcore: npm i -g @aws/agentcore"

# Images build in CodeBuild unless AGENT_BUILD=local, which builds the agent with this
# host's Docker — then a stopped daemon would fail the deploy minutes in rather than here.
if [ "$AGENT_BUILD" = "local" ]; then
  docker info >/dev/null 2>&1 || fail "Docker isn't running — AGENT_BUILD=local builds the agent image with it."
fi

# --- credentials ---------------------------------------------------------
IDENT="$(aws sts get-caller-identity --output json 2>/dev/null)" \
  || fail "AWS credentials not usable${PROFILE:+ for profile '$PROFILE'} — check .env, or run aws configure."
ACCOUNT="$(printf '%s' "$IDENT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Account"])')"

# The agent can't run without the model, and access is per-region opt-in.
# An unset MODEL_ID is reported with the rest of .env below.
if [ -n "${MODEL_ID:-}" ]; then
  aws bedrock get-foundation-model --model-identifier "${MODEL_ID#global.}" >/dev/null 2>&1 \
    || printf '\033[1;33m%s\033[0m\n' "  warning: can't confirm $MODEL_ID in $REGION — enable model access if the agent 403s"
fi

# --- permissions ---------------------------------------------------------
# One harmless read per service the deploy writes to. This catches the case that
# actually bites — a policy or SCP that doesn't reach the service at all — without
# the false confidence of SimulatePrincipalPolicy, which can't see SCPs or
# permission boundaries and needs its own permission to call. A read passing does
# not prove the matching write is allowed, so these only warn.
denied=""
probe() {  # label, command…
  local label="$1"; shift
  "$@" >/dev/null 2>&1 || denied="$denied $label"
}
probe cloudformation aws cloudformation list-stacks --max-items 1
probe agentcore      aws bedrock-agentcore-control list-agent-runtimes --max-results 1
probe secretsmanager aws secretsmanager list-secrets --max-items 1
probe dynamodb       aws dynamodb list-tables --max-items 1
probe ecr            aws ecr describe-repositories --max-items 1
probe codebuild      aws codebuild list-projects
probe cognito        aws cognito-idp list-user-pools --max-results 1
probe iam            aws iam list-roles --max-items 1
[ "$WEB_SEARCH" != "true" ] || probe "agentcore(us-east-1)" \
  aws bedrock-agentcore-control list-gateways --region us-east-1 --max-results 1
# Files storage reaches three services nothing else here touches, and s3files is new
# enough that an old CLI simply has no such command — which is worth catching before a
# deploy that creates a VPC.
if [ "${FILES_STORAGE:-false}" = "true" ]; then
  # boto3, not `aws s3files`: that subcommand is absent from AWS CLI 2.34, so probing it
  # through the CLI reports a denial where the real problem is the CLI's vocabulary.
  probe s3files uv run --with boto3 python -c \
    "import boto3; boto3.client('s3files').list_file_systems(maxResults=1)"
  probe kms     aws kms list-keys --limit 1
  probe ec2     aws ec2 describe-vpcs --max-items 1
fi
[ -z "$denied" ] || printf '\033[1;33m%s\033[0m\n' "  warning: no read access to:$denied
  The deploy writes to these — expect it to fail unless that is just a read-only denial."

# --- config --------------------------------------------------------------
[ -f .env ] || fail ".env is missing — start from: cp .env.example .env"
[ -n "${MODEL_ID:-}" ] || fail "unset in .env: MODEL_ID"
# Lark is all or nothing: half a Lark app deploys a bot that can never answer.
lark_keys="LARK_APP_ID LARK_APP_SECRET LARK_ENCRYPT_KEY LARK_VERIFICATION_TOKEN LARK_ADMIN_OPEN_ID"
set_keys="" blank=""
for k in $lark_keys; do
  if [ -n "${!k:-}" ]; then set_keys="$set_keys $k"; else blank="$blank $k"; fi
done
if [ -n "$set_keys" ] && [ -n "$blank" ]; then
  fail "Lark is half configured — unset in .env:$blank
  Fill them from the Lark developer console (README → Lark console setup), or clear all five to deploy without Lark."
fi
if [ "$LARK" != "true" ]; then
  # Without Lark the web page is the only way in, and someone has to be able to sign in.
  [ "${WEBUI:-false}" = "true" ] || fail "no Lark app and WEBUI is not true — nobody could reach the agent"
  [ -n "${WEB_ADMIN_EMAIL:-}" ] || fail "no Lark app and WEB_ADMIN_EMAIL is unset — nobody could sign in to the web page"
  [ -z "${AGENT_DECIDE_APPROVAL_CODES:-}" ] || fail "AGENT_DECIDE_APPROVAL_CODES needs Lark (approvals are Lark's)"
fi

echo "  deploy  : $DEPLOYMENT_NAME   (state: $STATE_FILE)"
echo "  account : $ACCOUNT"
echo "  region  : $REGION${PROFILE:+   profile: $PROFILE}"
echo "  model   : $MODEL_ID"
echo "  lark    : $([ "$LARK" = "true" ] && echo "$LARK_API_DOMAIN" || echo 'off  (web page only)')"
echo "  build   : $AGENT_BUILD$([ "$AGENT_BUILD" = "local" ] && echo '  (local Docker)' || echo '  (CodeBuild, ARM64 native)')"
echo "  search  : $WEB_SEARCH$([ "$WEB_SEARCH" = "true" ] && echo '  (+ a gateway in us-east-1)')"
echo "  files   : ${FILES_STORAGE:-false}$([ "${FILES_STORAGE:-false}" = "true" ] && echo "  (creates a VPC + S3 Files)" || true)"
if [ "${FILES_STORAGE:-false}" = "true" ]; then
  echo "  nat     : ${NAT_GATEWAY:-false}$([ "${NAT_GATEWAY:-false}" = "true" ] && echo '  (code sandbox gets internet; NAT is a fixed hourly cost)' || true)"
fi
