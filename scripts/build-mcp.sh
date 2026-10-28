#!/usr/bin/env bash
# Build the MCP servers under mcp-servers/ natively on ARM64 via CodeBuild, push to
# ECR, and create/update a Runtime for each.
#
# Usage: [PROFILE=p REGION=r] scripts/build-mcp.sh [name...]
#   no args   every directory under mcp-servers/ (subject to its DEPLOY_IF)
#   name...   only those, by directory name — DEPLOY_IF is ignored, since naming a
#             server is an explicit request and should not be second-guessed
#
# Each server describes itself in mcp-servers/<name>/runtime.env (runtime suffix,
# required vars, container env, optional DEPLOY_IF gate), so adding a server means
# adding a directory — not editing this script.
#
# Images are built natively on ARM64 in CodeBuild (scripts/lib/codebuild.sh).
set -euo pipefail
# Parsed whole before any of it runs: bash otherwise reads a script as it goes, so an edit
# made while this runs (an editor, another session) shifts what it reads next — measured as
# a syntax error halfway through a deploy.
{

. "$(cd "$(dirname "$0")" && pwd)/lib/env.sh"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
LARK_CLI_VERSION="${LARK_CLI_VERSION:-1.0.86}"   # engine = official lark-cli (not lark-mcp)

. scripts/lib/codebuild.sh

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }

# Which servers to handle. Explicit names skip the DEPLOY_IF gate.
EXPLICIT=0
if [ $# -gt 0 ]; then
  EXPLICIT=1
  SERVERS=("$@")
  for n in "${SERVERS[@]}"; do
    [ -f "mcp-servers/$n/runtime.env" ] || { echo "unknown MCP server: $n (expected mcp-servers/$n/runtime.env)"; exit 1; }
  done
else
  SERVERS=()
  for d in mcp-servers/*/; do
    [ -f "$d/runtime.env" ] && SERVERS+=("$(basename "$d")")
  done
fi
[ ${#SERVERS[@]} -gt 0 ] || { echo "no MCP servers found under mcp-servers/"; exit 1; }

# ---- per server ----------------------------------------------------------
for SERVER in "${SERVERS[@]}"; do
DIR="mcp-servers/$SERVER"
# Reset per-iteration so one server's config cannot leak into the next.
RUNTIME_SUFFIX=""; DEPLOY_IF=""; REQUIRE_VARS=""; RUNTIME_ENV_MAP=""; IMAGE_TAG=""
. "$DIR/runtime.env"
: "${RUNTIME_SUFFIX:?$DIR/runtime.env must set RUNTIME_SUFFIX}"

# A server can declare a variable that gates it. Skipped only for a build-everything
# run — an explicitly named server is built regardless.
if [ "$EXPLICIT" = "0" ] && [ -n "$DEPLOY_IF" ] && [ -z "${!DEPLOY_IF:-}" ]; then
  warn "skipping $SERVER — $DEPLOY_IF is unset (name it explicitly to build anyway)"
  continue
fi
for v in $REQUIRE_VARS; do
  [ -n "${!v:-}" ] || { echo "$SERVER needs $v — set it in .env"; exit 1; }
done

# The tag comes from the server itself (IMAGE_TAG in its runtime.env), so a server
# that doesn't wrap lark-cli isn't labelled with a CLI version it never installs.
# SRC_SHA lets such a server tag by content, which also makes "did this deploy
# actually change anything?" visible at a glance in ECR.
# The build context is the server's directory plus the protocol code they all share, so a
# change to either rebuilds the image.
CTX="$(mktemp -d)"
cp -R "$DIR"/. "$CTX"/
cp mcp-servers/_shared/mcp-protocol.js "$CTX"/
# shellcheck disable=SC2034  # read by the eval of IMAGE_TAG below
SRC_SHA="$(cd "$CTX" && find . -type f -exec sha256sum {} + | sort | sha256sum | cut -c1-12)"
TAG="$(eval echo "${IMAGE_TAG:-cli-\$LARK_CLI_VERSION}")"

REPO="$PREFIX-mcp-$SERVER"
ECR="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO"
RUNTIME_NAME="${PREFIX//-/_}_$RUNTIME_SUFFIX"   # AgentCore runtime names use underscores
printf '\n\033[1;36m### MCP server: %s → %s\033[0m\n' "$SERVER" "$RUNTIME_NAME"

log "Build $REPO:$TAG (CodeBuild, ARM64 native)"
cb_build "$REPO" "$CTX" "$TAG" --build-arg "LARK_CLI_VERSION=$LARK_CLI_VERSION"
rm -rf "$CTX"  # safe-rm-ok

log "MCP server Runtime ($RUNTIME_NAME)"
# The per-user Lark token reaches the container in a custom passthrough header —
# it is stripped at the edge unless listed in requestHeaderAllowlist.
ROLE_ARN="$(aws cloudformation describe-stacks --stack-name "$PREFIX-agentcore" \
  --query "Stacks[0].Outputs[?OutputKey=='ExecutionRoleArn'].OutputValue" --output text 2>/dev/null)"
[ -n "$ROLE_ARN" ] && [ "$ROLE_ARN" != "None" ] || {
  echo "execution role not found — run ./deploy.sh base first"; exit 1; }

ARTIFACT="{\"containerConfiguration\":{\"containerUri\":\"$ECR:$TAG\"}}"
HEADERS='{"requestHeaderAllowlist":["X-Amzn-Bedrock-AgentCore-Runtime-Custom-Lark-Token"]}'
# Container env from the server's RUNTIME_ENV_MAP ("CONTAINER_KEY=SOURCE_VAR" pairs).
# Emitted as JSON because a value may contain commas (the approval allow-list does),
# and the CLI's key=value shorthand splits on those and mangles the whole map.
ENVVARS="$(MAP="$RUNTIME_ENV_MAP" uv run python -c '
import json, os
out = {}
for pair in os.environ["MAP"].split():
    key, _, src = pair.partition("=")
    out[key] = os.environ.get(src or key, "")
print(json.dumps(out))')"

RID="$(aws bedrock-agentcore-control list-agent-runtimes \
  --query "agentRuntimes[?agentRuntimeName=='$RUNTIME_NAME'].agentRuntimeId" \
  --output text 2>/dev/null | head -1)"
if [ -n "$RID" ] && [ "$RID" != "None" ]; then
  aws bedrock-agentcore-control update-agent-runtime --agent-runtime-id "$RID" \
    --agent-runtime-artifact "$ARTIFACT" --role-arn "$ROLE_ARN" \
    --network-configuration '{"networkMode":"PUBLIC"}' \
    --protocol-configuration '{"serverProtocol":"MCP"}' \
    --environment-variables "$ENVVARS" \
    --request-header-configuration "$HEADERS" >/dev/null
  echo "  updated $RUNTIME_NAME ($RID)"
else
  RID="$(aws bedrock-agentcore-control create-agent-runtime --agent-runtime-name "$RUNTIME_NAME" \
    --agent-runtime-artifact "$ARTIFACT" --role-arn "$ROLE_ARN" \
    --network-configuration '{"networkMode":"PUBLIC"}' \
    --protocol-configuration '{"serverProtocol":"MCP"}' \
    --environment-variables "$ENVVARS" \
    --request-header-configuration "$HEADERS" \
    --query agentRuntimeId --output text)"
  echo "  created $RUNTIME_NAME ($RID)"
fi

log "Done"
echo "IMAGE_URI=$ECR:$TAG"
echo "  runtime: $RUNTIME_NAME ($RID)"

done
exit
}
