#!/usr/bin/env bash
# Deploy the agent. Override the target with PROFILE=... REGION=... env vars.
#
# Steps (idempotent — re-runnable independently):
#   --base      CDK base stacks (security, agentcore, router, gateway, observability)
#   --memory    AgentCore Memory for long-term records (implied by --runtime)
#   --runtime   create/update the AgentCore Runtime from the built image (CLI)
#   --gateway   Web Search gateway in us-east-1 (only when WEB_SEARCH=true)
#   --code      Code Interpreter for code generation/execution (FILES_STORAGE=true)
#   --webui     static web chat on S3 + CloudFront (only when WEBUI=true)
#   (no arg)    run all steps in order
#
# Step implementation — normally invoked through ./deploy.sh in the repo root,
# which owns the ordering. Callable directly when iterating on one phase:
# Usage: [PROFILE=p REGION=r] scripts/provision.sh [--base|--memory|--code|--runtime|--gateway|--a2a|--webui]
set -euo pipefail
# Parsed whole before any of it runs: bash otherwise reads a script as it goes, so an edit
# made while this runs (an editor, another session) shifts what it reads next — measured as
# a syntax error halfway through a deploy.
{

. "$(cd "$(dirname "$0")" && pwd)/lib/env.sh"
WEB_SEARCH="${WEB_SEARCH:-false}"
# Code execution and its per-user workspace. Off by default: it adds a VPC (the sandbox
# reaches S3 via a free gateway endpoint) and an S3 Files file system. NAT_GATEWAY adds a
# NAT to that VPC so the sandbox has internet; it means nothing without FILES_STORAGE.
# See docs/architecture.md.
FILES_STORAGE="${FILES_STORAGE:-false}"
NAT_GATEWAY="${NAT_GATEWAY:-false}"
[ "$FILES_STORAGE" = "true" ] || NAT_GATEWAY=false
# The agent Runtime's network, fixed for the deployment's life. vpc without AGENT_SUBNET_IDS
# creates the network stack; with them the agent goes into the operator's subnets.
. scripts/lib/network.sh
network_check_lock
NETWORK_STACK="$([ "$NETWORK_KIND" = "vpc" ] && echo true || echo false)"
# Where the agent image is built: CodeBuild (ARM64 native, no local Docker) unless .env
# asks for the host's Docker.
AGENT_BUILD="${AGENT_BUILD:-codebuild}"
case "$AGENT_BUILD" in codebuild|local) ;; *) echo "AGENT_BUILD must be codebuild or local, not '$AGENT_BUILD'"; exit 1 ;; esac
export UV_LINK_MODE=copy

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
export CDK_DEFAULT_ACCOUNT="$ACCOUNT" CDK_DEFAULT_REGION="$REGION"
CDK="npx --yes aws-cdk@2"
: "${MODEL_ID:?set MODEL_ID in .env}"
: "${LARK_API_DOMAIN:?set LARK_API_DOMAIN in .env}"
# Every cdk deploy gets the same context: a deploy that omitted one would synth that stack
# with the switch off and tear the resources out.
CDK_CTX=(-c "deployment_name=$PREFIX" -c "files_storage=$FILES_STORAGE" -c "nat_gateway=$NAT_GATEWAY"
         -c "agent_build=$AGENT_BUILD" -c "lark_api_domain=$LARK_API_DOMAIN"
         -c "webui=${WEBUI:-false}" -c "lark=$([ -n "${LARK_APP_ID:-}" ] && echo true || echo false)"
         -c "agent_network=$NETWORK_KIND" --output "$CDK_OUT")
. scripts/lib/codebuild.sh

# Control-plane calls whose parameters are newer than the AWS CLI's bundled service model.
# Verified against CLI 2.34: it rejects requireServiceS3Endpoint, sessionConfiguration and
# clientAuthenticationMethod as unknown parameters even though the service accepts them.
# uv resolves a current botocore, so these calls do not depend on how fresh the operator's
# CLI happens to be. Usage: acp <operation_name> <json params>  → prints the JSON result.
acp() {
  ACP_OP="$1" ACP_PARAMS="$2" ACP_REGION="${3:-$REGION}" \
  uv run --with 'boto3>=1.43.92' python - <<'PYEOF'
import json, os, sys
import boto3
c = boto3.client("bedrock-agentcore-control", region_name=os.environ["ACP_REGION"])
try:
    r = getattr(c, os.environ["ACP_OP"])(**json.loads(os.environ["ACP_PARAMS"]))
except Exception as e:
    sys.stderr.write(f"{type(e).__name__}: {e}\n")
    sys.exit(1)
print(json.dumps({k: v for k, v in r.items() if k != "ResponseMetadata"}, default=str))
PYEOF
}

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

cfn_out() { # stack, output-key
  aws cloudformation describe-stacks --stack-name "$1" \
    --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text 2>/dev/null
}

ctx_get() { # key — read a deployment id back out of $STATE_FILE ("" when absent)
  STATE_FILE="$STATE_FILE" uv run python - "$1" <<'PYGET'
import json, os, sys
f = os.environ["STATE_FILE"]
print((json.load(open(f)) if os.path.isfile(f) else {}).get(sys.argv[1], ""))
PYGET
}

ctx_set() { # key value  — persist a deployment id into $STATE_FILE (gitignored)
  STATE_FILE="$STATE_FILE" uv run python - "$1" "$2" <<'PY'
import json, os, sys
k, v = sys.argv[1], sys.argv[2]
f = os.environ["STATE_FILE"]
d = json.load(open(f)) if os.path.isfile(f) else {}
d[k] = v
with open(f, "w") as fh: json.dump(d, fh, indent=2); fh.write("\n")
print(f"{f}: {k} = {v}")
PY
}

# The agent image the Runtimes run, in AGENT_IMAGE. Under CodeBuild it is tagged by the
# content of what the Dockerfile copies, so an unchanged agent is not rebuilt.
ensure_agent_image() {
  if [ "$AGENT_BUILD" = "local" ]; then
    AGENT_IMAGE="$(cfn_out "$PREFIX-agentcore" AgentImageUri)"
    [ -n "$AGENT_IMAGE" ] && [ "$AGENT_IMAGE" != "None" ] || {
      echo "missing agent image output — run --base with AGENT_BUILD=local first"; exit 1; }
    return 0
  fi
  local repo="$PREFIX-agent" tag
  tag="src-$(python3 - <<'PYHASH'
import glob, hashlib, os
os.chdir("agent")
h = hashlib.sha256()
for f in sorted({"Dockerfile", "requirements.txt", "entrypoint.sh",
                *glob.glob("**/*.py", recursive=True), *glob.glob("assets/*")}):
    h.update(f.encode() + b"\0" + open(f, "rb").read())
print(h.hexdigest()[:12])
PYHASH
)"
  if cb_image_exists "$repo" "$tag"; then
    echo "  agent image $repo:$tag already built — source unchanged"
    AGENT_IMAGE="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$repo:$tag"
  else
    log "Agent image — build $repo:$tag (CodeBuild, ARM64 native)"
    cb_build "$repo" agent "$tag"
    AGENT_IMAGE="$CB_IMAGE_URI"
    # Every source change pushes a new tag; keep the recent ones for rollback, not all.
    aws ecr put-lifecycle-policy --repository-name "$repo" --lifecycle-policy-text \
      '{"rules":[{"rulePriority":1,"description":"keep the last 10 images","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":10},"action":{"type":"expire"}}]}' >/dev/null
  fi
  ctx_set agent_image_uri "$AGENT_IMAGE"
}

base_cdk_stacks() {
  # First deploy in a region needs the CDK bootstrap stack; it creates an assets
  # bucket/ECR repo and deploy roles, so this step needs broader IAM permissions.
  aws cloudformation describe-stacks --stack-name CDKToolkit >/dev/null 2>&1 || {
    log "CDK bootstrap — first deploy in $REGION"
    $CDK bootstrap "aws://$ACCOUNT/$REGION"
  }

  log "Base — CDK stacks"
  local stacks=("$PREFIX-security" "$PREFIX-agentcore" "$PREFIX-router"
                "$PREFIX-gateway" "$PREFIX-shim" "$PREFIX-observability")
  ctx_set agent_network "$NETWORK_KIND"
  # Kept, so dropping it from .env later does not quietly move the NAT to a new address.
  [ -z "${EIP_ALLOCATION_ID:-}" ] || ctx_set eip_allocation_id "$EIP_ALLOCATION_ID"
  if [ "$NETWORK_STACK" = "true" ]; then
    # Resolved once: re-resolving on a later deploy could pick other AZs and move the subnets.
    if [ -z "$(ctx_get network_azs)" ]; then
      local azs
      azs="$(supported_az_names)"
      [ "$(echo "$azs" | tr ',' '\n' | grep -c .)" -ge 2 ] || {
        echo "no two AgentCore-supported AZs in $REGION for this account — see docs/deploy.md"; exit 1; }
      ctx_set network_azs "$azs"
    fi
    stacks+=("$PREFIX-network")
    echo "  agent network: VPC ($(ctx_get network_azs); NAT + flow logs)"
  elif [ "$AGENT_NETWORK" = "vpc" ]; then
    echo "  agent network: VPC (your subnets $AGENT_SUBNET_IDS)"
  fi
  # The sandbox stack only exists when the flag is on (app.py skips it otherwise), so
  # naming it unconditionally would fail the deploy rather than skip it.
  if [ "$FILES_STORAGE" = "true" ]; then
    stacks+=("$PREFIX-sandbox")
    if [ "$NAT_GATEWAY" = "true" ]; then
      echo "  code execution: on (VPC with a NAT gateway + S3 Files; the sandbox has internet)"
    else
      echo "  code execution: on (VPC without NAT + S3 Files; the sandbox has no internet)"
    fi
  else
    echo "  code execution: off"
  fi
  $CDK deploy "${stacks[@]}" "${CDK_CTX[@]}" \
             --require-approval never --outputs-file "$CDK_OUT/outputs.json"
  # The web page needs this and CDK assigns it, so it travels through state rather than a
  # cross-stack export — importing it would freeze the router's own exports.
  ctx_set router_api_url "$(cfn_out "$PREFIX-router" ApiUrl)"
}

phase_memory() {
  log "Memory — long-term store"

  # Long-term records only; the conversation lives in DynamoDB. No memoryStrategies on
  # purpose — direct writes are embedded and scored without one, and a custom strategy is
  # rejected without an execution role. eventExpiryDuration is required. See docs/architecture.md.
  local mname="${PREFIX//-/_}_ltm" mid
  # list-memories returns no `name` — only `id`, which is "<name>-<suffix>". Matching on
  # name would never hit, and this step would create a new memory on every run.
  mid="$(aws bedrock-agentcore-control list-memories --max-results 100 \
        --query "memories[?starts_with(id, '${mname}-')].id" --output text 2>/dev/null | head -1)"

  if [ -n "$mid" ] && [ "$mid" != "None" ]; then
    echo "  reusing $mid"
  else
    mid="$(acp create_memory "$(MNAME="$mname" uv run python -c '
import json, os
print(json.dumps({
    "name": os.environ["MNAME"],
    "description": "Long-term records for the agent; the conversation lives in DynamoDB",
    "eventExpiryDuration": 365,
}))')" | uv run python -c 'import json,sys;print(json.load(sys.stdin)["memory"]["id"])')"
    echo "  created $mid"
  fi

  # Wait for ACTIVE: creation is asynchronous and a write to a CREATING memory fails
  # rather than queueing.
  for _ in $(seq 1 40); do
    [ "$(aws bedrock-agentcore-control get-memory --memory-id "$mid" \
         --query 'memory.status' --output text 2>/dev/null)" = "ACTIVE" ] && break
    sleep 5
  done
  ctx_set memory_id "$mid"
}


phase2_runtime() {
  # The agent Runtime is created straight from the control plane, from the ARM64 image
  # ensure_agent_image built (CodeBuild) or CDK published (AGENT_BUILD=local).
  #
  # The AgentCore CLI used to do this (`agentcore configure` + `agentcore deploy`), but
  # 0.28 removed `configure` and turned `deploy` into "deploy project infrastructure via
  # CDK", which wants to own the CDK app this repo already has. Driving the API directly
  # removes the dependency on a fast-moving CLI and matches what build-mcp.sh has always
  # done for the MCP server runtimes. It also lets the authorizer and network mode be set
  # at creation time instead of patched in afterwards.
  ensure_agent_image
  log "Runtime — create/update the agent Runtime from $AGENT_IMAGE"

  local role image model memory shim mcp_arn mcp_url pool client webclient ws_url
  local approval_url approval_arn issuer ckpt_table ckpt_bucket google_url google_arn
  role="$(cfn_out "$PREFIX-agentcore" ExecutionRoleArn)"
  image="$AGENT_IMAGE"
  ckpt_table="$(cfn_out "$PREFIX-agentcore" CheckpointTableName)"
  ckpt_bucket="$(cfn_out "$PREFIX-agentcore" CheckpointBucketName)"
  pool="$(cfn_out "$PREFIX-security" UserPoolId)"
  client="$(cfn_out "$PREFIX-security" UserPoolClientId)"
  # The hosted login's client, when the web page is deployed: its tokens are web users'.
  webclient="$(cfn_out "$PREFIX-security" WebClientId)"
  if [ "$webclient" = "None" ]; then webclient=""; fi
  issuer="$(cfn_out "$PREFIX-security" CognitoIssuerUrl)"
  ws_url="$(ctx_get websearch_gateway_url)"
  model="$MODEL_ID"
  memory="$(ctx_get memory_id)"
  shim="$(cfn_out "$PREFIX-shim" ShimReturnUrl)"

  mcp_arn="$(aws bedrock-agentcore-control list-agent-runtimes \
    --query "agentRuntimes[?agentRuntimeName=='${PREFIX//-/_}_mcp'].agentRuntimeArn" --output text 2>/dev/null | head -1)"
  mcp_url="https://bedrock-agentcore.$REGION.amazonaws.com/runtimes/$(uv run python -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$mcp_arn")/invocations?qualifier=DEFAULT"
  approval_arn="$(aws bedrock-agentcore-control list-agent-runtimes \
    --query "agentRuntimes[?agentRuntimeName=='${PREFIX//-/_}_approval'].agentRuntimeArn" --output text 2>/dev/null | head -1)"
  if [ -n "$approval_arn" ] && [ "$approval_arn" != "None" ]; then
    approval_url="https://bedrock-agentcore.$REGION.amazonaws.com/runtimes/$(uv run python -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$approval_arn")/invocations?qualifier=DEFAULT"
    echo "  approval tools: enabled"
  else
    approval_url=""
    echo "  approval tools: not deployed (./deploy.sh mcp approval to add them)"
  fi

  # The second downstream system, discovered the same way: deployed or not, nothing else
  # changes. Its provider name is fixed by setup-3lo.sh.
  google_arn="$(aws bedrock-agentcore-control list-agent-runtimes \
    --query "agentRuntimes[?agentRuntimeName=='${PREFIX//-/_}_google'].agentRuntimeArn" --output text 2>/dev/null | head -1)"
  if [ -n "$google_arn" ] && [ "$google_arn" != "None" ]; then
    google_url="https://bedrock-agentcore.$REGION.amazonaws.com/runtimes/$(uv run python -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$google_arn")/invocations?qualifier=DEFAULT"
    echo "  google tools: enabled"
  else
    google_url=""
    echo "  google tools: not deployed (./deploy.sh mcp google to add them)"
  fi

  [ -n "$role" ] || { echo "missing execution role output — run --base first"; exit 1; }
  [ -n "$image" ] || { echo "missing agent image output — run --base first"; exit 1; }
  # Not fatal: the agent falls back to in-process state and keeps answering. Loud, because
  # the symptom is "the bot forgot everything after a timeout", which reads as a bug.
  [ -n "$ckpt_table" ] && [ "$ckpt_table" != "None" ] || \
    echo "  WARNING: no checkpoint table output — conversations will not survive a new microVM"
  if [ "$LARK" = "true" ]; then
    [ -n "$mcp_arn" ] && [ "$mcp_arn" != "None" ] || {
      echo "${PREFIX//-/_}_mcp runtime not found — run ./deploy.sh mcp first"; exit 1; }
  else
    mcp_url=""
    echo "  lark tools: off (no LARK_APP_ID)"
  fi

  # Code execution: the Code Interpreter session joins the sandbox subnets, and only the ids
  # it needs are baked in here. See docs/architecture.md.
  local fs_arn="" ci_id=""
  if [ "$FILES_STORAGE" = "true" ]; then
    fs_arn="$(cfn_out "$PREFIX-sandbox" FileSystemArn)"
    ci_id="$(ctx_get code_interpreter_id)"
    [ -n "$ci_id" ] || { echo "no code interpreter — run --code first"; exit 1; }
    echo "  code execution: on (interpreter $ci_id)"
  fi

  local net_subnets="" net_sgs=""
  if [ "$AGENT_NETWORK" = "vpc" ]; then
    if [ "$NETWORK_KIND" = "vpc:byo" ]; then
      net_subnets="$AGENT_SUBNET_IDS" net_sgs="$AGENT_SECURITY_GROUP_IDS"
    else
      net_subnets="$(cfn_out "$PREFIX-network" AgentSubnetIds)"
      net_sgs="$(cfn_out "$PREFIX-network" AgentSecurityGroupId)"
    fi
    [ -n "$net_subnets" ] && [ "$net_subnets" != "None" ] && [ -n "$net_sgs" ] && [ "$net_sgs" != "None" ] || {
      echo "no subnets or security group for the agent — run --base first"; exit 1; }
    echo "  network: VPC (subnets $net_subnets)"
  fi

  local rname="${PREFIX//-/_}_agent" params rid
  params="$(RNAME="$rname" IMAGE="$image" ROLE="$role" ISSUER="$issuer" CLIENT="$client" \
    NET_SUBNETS="$net_subnets" NET_SGS="$net_sgs" \
    WEBCLIENT="$webclient" \
    MODEL="$model" MEMORY="$memory" MCP_URL="$mcp_url" \
    SHIM="$shim" POOL="$pool" WS_URL="$ws_url" \
    APPROVAL_URL="$approval_url" FS_ARN="$fs_arn" CI_ID="$ci_id" PREFIX="$PREFIX" \
    CODE_NET="$([ -n "$ci_id" ] && [ "$NAT_GATEWAY" = "true" ] && echo true || true)" \
    GOOGLE_URL="$google_url" \
    CKPT_TABLE="$ckpt_table" CKPT_BUCKET="$ckpt_bucket" \
    LARK_DOMAIN="$LARK_API_DOMAIN" HAS_LARK="${LARK_APP_ID:+1}" \
    HAS_GITHUB="$([ -n "${GITHUB_CLIENT_ID:-}" ] && [ -n "${GITHUB_CLIENT_SECRET:-}" ] && echo 1 || true)" \
    uv run python - <<'PYEOF'
import json, os
e = os.environ
# Without a Lark app these stay empty, which is how the agent knows it has no Lark.
lark = bool(e.get("HAS_LARK"))
platform_version = "V2"
env = {
    "BEDROCK_MODEL_ID": e["MODEL"],
    # Not visible to the container any other way — platformVersion is a control-plane
    # field on the Runtime resource, not something AgentCore injects at runtime.
    "AGENT_PLATFORM_VERSION": platform_version,
    # Conversation state. Without the table the agent still answers, but history dies
    # with the container — so a missing output here is a real regression, not an option.
    "CHECKPOINT_TABLE": e["CKPT_TABLE"],
    # Tunable without a code change: the right threshold depends on the model's context
    # window and on how much a cache-prefix invalidation costs at your traffic.
    "SUMMARIZE_AT_TOKENS": os.environ.get("SUMMARIZE_AT_TOKENS", ""),
    "PROMPT_CACHE_TTL": os.environ.get("PROMPT_CACHE_TTL", ""),
    # Empty means "keep the default in agent/core.py"; the env filter below drops it.
    "AGENT_SYSTEM_PROMPT": os.environ.get("AGENT_SYSTEM_PROMPT", ""),
    "CHECKPOINT_BUCKET": e["CKPT_BUCKET"],
    # Long-term memory only; empty until the Memory resource exists.
    "BEDROCK_AGENTCORE_MEMORY_ID": e["MEMORY"],
    "LARK_MCP_URL": e["MCP_URL"],
    "SHIM_RETURN_URL": e["SHIM"],
    "LARK_OAUTH_PROVIDER": e["PREFIX"] + "-3lo" if lark else "",
    "AGENT_WORKLOAD_NAME": e["PREFIX"] + "-wl",
    # The execution role may write metrics only to this namespace (agentcore stack).
    "METRICS_NAMESPACE": e["PREFIX"] + "/agent",
    "LARK_SCOPES": "drive:drive docx:document offline_access" if lark else "",
    "LARK_SECRET_ID": e["PREFIX"] + "/channels/lark" if lark else "",
    "LARK_API_DOMAIN": e["LARK_DOMAIN"],
    "COGNITO_USER_POOL_ID": e["POOL"],
    "COGNITO_CLIENT_ID": e["CLIENT"],
    "COGNITO_WEB_CLIENT_ID": e["WEBCLIENT"],
    "WEBSEARCH_GATEWAY_URL": e["WS_URL"],
    "APPROVAL_MCP_URL": e["APPROVAL_URL"],
    # Both empty unless the google server is deployed; the agent then has no Google tools.
    "GOOGLE_MCP_URL": e["GOOGLE_URL"],
    "GOOGLE_OAUTH_PROVIDER": (e["PREFIX"] + "-google") if e["GOOGLE_URL"] else "",
    "GOOGLE_SCOPES": os.environ.get("GOOGLE_SCOPES", ""),
    # GitHub's tools are GitHub's own remote MCP server: a provider is all it takes.
    "GITHUB_OAUTH_PROVIDER": (e["PREFIX"] + "-github") if e.get("HAS_GITHUB") else "",
    "GITHUB_MCP_TOOLSETS": os.environ.get("GITHUB_MCP_TOOLSETS", ""),
    "GITHUB_MCP_READONLY": os.environ.get("GITHUB_MCP_READONLY", ""),
    # Empty unless code execution is on; the agent treats that as "no code tools".
    "CODE_INTERPRETER_ID": e["CI_ID"],
    "FILES_FS_ARN": e["FS_ARN"],
    "WORKSPACE_PATH": "/mnt/workspace" if e["CI_ID"] else "",
    # Tells the code tools whether to promise installs; the NAT is what makes it true.
    "CODE_INTERNET": e["CODE_NET"],
    # The browser tools; the role's grant on aws.browser.v1 is unconditional.
    "BROWSER_ENABLED": os.environ.get("BROWSER", ""),
}
# PUBLIC unless AGENT_NETWORK=vpc. In a VPC every call the agent makes leaves through those
# subnets, so they must route to the internet: Lark, Google and the us-east-1 Gateway are
# outside it. No requireServiceS3Endpoint: refused at creation (see the update path below).
net = {"networkMode": "PUBLIC"}
if e.get("NET_SUBNETS"):
    net = {"networkMode": "VPC", "networkModeConfig": {
        "subnets": [x for x in e["NET_SUBNETS"].split(",") if x],
        "securityGroups": [x for x in e["NET_SGS"].split(",") if x]}}
print(json.dumps({
    "agentRuntimeName": e["RNAME"],
    # V2 here demonstrates the platform version; what it costs and gains against V1 is measured
    # in docs/agentcore-behavior.md. Three behaviours were checked before enabling it:
    # session-to-instance affinity still holds, so the in-process turn lock stays valid; the
    # module-scope event loop and its MCP sessions survive a restore; and anything derived
    # at import is snapshot-shared, which is why server.py takes its instance identity on
    # the first request. Creation now takes 120-190 s rather than seconds.
    "platformVersion": platform_version,
    "agentRuntimeArtifact": {"containerConfiguration": {"containerUri": e["IMAGE"]}},
    "roleArn": e["ROLE"],
    "networkConfiguration": net,
    "protocolConfiguration": {"serverProtocol": "HTTP"},
    # The agent verifies the caller's token itself and takes the actor from it (caller.py);
    # Authorization is stripped at the edge unless listed.
    "requestHeaderConfiguration": {"requestHeaderAllowlist": ["Authorization"]},
    "environmentVariables": {k: v for k, v in env.items() if v},
    # Inbound CUSTOM_JWT: the platform verifies the caller and hands the agent a workload
    # token derived from that identity, so the agent never names a user itself.
    "authorizerConfiguration": {"customJWTAuthorizer": {
        "discoveryUrl": e["ISSUER"] + "/.well-known/openid-configuration",
        "allowedClients": [c for c in (e["CLIENT"], e.get("WEBCLIENT", "")) if c]}},
}))
PYEOF
)"

  rid="$(aws bedrock-agentcore-control list-agent-runtimes \
    --query "agentRuntimes[?agentRuntimeName=='$rname'].agentRuntimeId" --output text 2>/dev/null | head -1)"
  if [ -n "$rid" ] && [ "$rid" != "None" ]; then
    # update takes the id instead of the name, and replaces rather than patches — which is
    # why every field above is sent every time.
    acp update_agent_runtime "$(RID="$rid" P="$params" uv run python -c '
import json, os
p = json.loads(os.environ["P"]); p.pop("agentRuntimeName", None)
p["agentRuntimeId"] = os.environ["RID"]
# No requireServiceS3Endpoint: rejected at creation, and "agents created after
# 2026-06-08 cannot modify requireServiceS3Endpoint" on update — so for anything built
# now it is unreachable, not update-only as an earlier reading of the error suggested.
# Leaving it in broke every runtime update while the first deploy still succeeded,
# because only the update path sent it.
print(json.dumps(p))')" >/dev/null
    echo "  updated $rname ($rid)"
  else
    rid="$(acp create_agent_runtime "$params" | uv run python -c 'import json,sys;print(json.load(sys.stdin)["agentRuntimeId"])')"
    echo "  created $rname ($rid)"
  fi
  ctx_set runtime_id "$rid"

  for _ in $(seq 1 40); do
    [ "$(aws bedrock-agentcore-control get-agent-runtime --agent-runtime-id "$rid" \
         --query status --output text 2>/dev/null)" = "READY" ] && break
    sleep 5
  done
  echo "  status: $(aws bedrock-agentcore-control get-agent-runtime --agent-runtime-id "$rid" --query status --output text 2>/dev/null)"

  # The router's AGENTCORE_RUNTIME_ARN was synthesised before the runtime existed (a
  # PLACEHOLDER), so re-deploy it now that the real id is known.
  log "Router — re-deploy with the real runtime ARN"
  $CDK deploy "$PREFIX-router" "${CDK_CTX[@]}" --require-approval never

  # AgentCore keeps serving existing sessions from the OLD container, so stored session
  # ids would pin users to the previous image. Drop them: the next message starts a new
  # session on the just-deployed version.
  log "Sessions — drop stored ids so users land on the new version"
  local n=0
  for pk in $(aws dynamodb scan --table-name "$PREFIX-identity" \
      --filter-expression "SK = :s" \
      --expression-attribute-values '{":s":{"S":"SESSION"}}' \
      --projection-expression "PK" --query 'Items[].PK.S' --output text 2>/dev/null); do
    aws dynamodb delete-item --table-name "$PREFIX-identity" \
      --key "{\"PK\":{\"S\":\"$pk\"},\"SK\":{\"S\":\"SESSION\"}}" >/dev/null 2>&1 && n=$((n+1))
  done
  echo "  dropped $n session(s)"
}

phase_code() {
  log "Code — the Code Interpreter the agent runs generated code in"
  if [ "$FILES_STORAGE" != "true" ]; then
    echo "  code execution: off (set FILES_STORAGE=true in .env)"; return 0
  fi
  local role subnets sg name existing
  role="$(cfn_out "$PREFIX-sandbox" CodeInterpreterRoleArn)"
  subnets="$(cfn_out "$PREFIX-sandbox" SandboxSubnetIds)"
  sg="$(cfn_out "$PREFIX-sandbox" SandboxSecurityGroupId)"
  [ -n "$subnets" ] && [ "$subnets" != "None" ] && [ -n "$role" ] && [ "$role" != "None" ] || {
    echo "sandbox stack outputs missing — run --base with FILES_STORAGE=true first"; exit 1; }
  name="${PREFIX//-/_}_code"

  # VPC mode, not SANDBOX: measured — filesystemConfigurations is refused outside VPC mode
  # ("VPC network mode is required when filesystemConfigurations are provided"). The mount
  # needs only the S3 gateway endpoint the sandbox stack creates; NAT_GATEWAY adds internet.
  #
  # No filesystemConfigurations here on purpose: the mount is named per session, so one
  # shared interpreter serves every user with their own Access Point. See docs/architecture.md.
  existing="$(aws bedrock-agentcore-control list-code-interpreters \
    --query "codeInterpreterSummaries[?name=='$name' && status!='DELETED' && status!='DELETING'].codeInterpreterId" \
    --output text 2>/dev/null | head -1)"
  if [ -n "$existing" ] && [ "$existing" != "None" ]; then
    # There is no UpdateCodeInterpreter, so an interpreter whose role, subnets or security
    # group no longer match is replaced. Files are unaffected (they live in S3 Files);
    # sessions open on the old one end.
    if aws bedrock-agentcore-control get-code-interpreter --code-interpreter-id "$existing" \
         --output json | ROLE="$role" SUBNETS="$subnets" SG="$sg" uv run python -c '
import json, os, sys
c = json.load(sys.stdin); v = c["networkConfiguration"].get("vpcConfig", {})
want = (os.environ["ROLE"], sorted(x for x in os.environ["SUBNETS"].split(",") if x), [os.environ["SG"]])
sys.exit(0 if (c.get("executionRoleArn"), sorted(v.get("subnets", [])), v.get("securityGroups", [])) == want else 1)'
    then
      echo "  reusing $existing"
      ctx_set code_interpreter_id "$existing"
      return 0
    fi
    echo "  $existing has a stale role or network config — replacing it"
    aws bedrock-agentcore-control delete-code-interpreter --code-interpreter-id "$existing" >/dev/null
    for _ in $(seq 1 40); do
      case "$(aws bedrock-agentcore-control get-code-interpreter --code-interpreter-id "$existing" \
                --query status --output text 2>/dev/null || echo GONE)" in
        GONE|DELETED) break ;;
      esac
      sleep 6
    done
  fi
  local cid
  cid="$(acp create_code_interpreter "$(NAME="$name" ROLE="$role" SUBNETS="$subnets" SG="$sg" \
    uv run python -c '
import json, os
e = os.environ
print(json.dumps({
    "name": e["NAME"],
    "description": "Runs generated code for one user at a time, with their own workspace mounted",
    "executionRoleArn": e["ROLE"],
    "networkConfiguration": {"networkMode": "VPC", "vpcConfig": {
        "subnets": [x for x in e["SUBNETS"].split(",") if x],
        "securityGroups": [e["SG"]],
        "requireServiceS3Endpoint": True}},
}))')" | uv run python -c 'import json,sys;print(json.load(sys.stdin)["codeInterpreterId"])')"
  echo "  created $cid"
  # READY is asynchronous; starting a session against a CREATING interpreter fails.
  for _ in $(seq 1 40); do
    [ "$(aws bedrock-agentcore-control get-code-interpreter --code-interpreter-id "$cid" \
         --query 'status' --output text 2>/dev/null)" = "READY" ] && break
    sleep 6
  done
  ctx_set code_interpreter_id "$cid"
}


phase_a2a() {
  log "A2A — expose the agent to peer agents"
  if [ "${A2A:-false}" != "true" ]; then
    echo "  a2a: off (set A2A=true in .env)"; return 0
  fi
  local role image rname rid params
  role="$(cfn_out "$PREFIX-agentcore" ExecutionRoleArn)"
  ensure_agent_image
  image="$AGENT_IMAGE"
  rname="${PREFIX//-/_}_a2a"

  # Same image as the agent, SERVER_MODE=a2a picks the other entrypoint. Only the identity
  # and model variables are needed: an A2A task runs a normal turn.
  params="$(RNAME="$rname" IMAGE="$image" ROLE="$role" PREFIX="$PREFIX" \
    ISSUER="$(cfn_out "$PREFIX-security" CognitoIssuerUrl)" \
    CLIENT="$(cfn_out "$PREFIX-security" UserPoolClientId)" \
    WEBCLIENT="$(cfn_out "$PREFIX-security" WebClientId | grep -v '^None$' || true)" \
    AGENT_ARN="$(aws bedrock-agentcore-control list-agent-runtimes \
      --query "agentRuntimes[?agentRuntimeName=='${PREFIX//-/_}_agent'].agentRuntimeArn" \
      --output text | head -1)" \
    SRC="$(aws bedrock-agentcore-control get-agent-runtime \
      --agent-runtime-id "$(aws bedrock-agentcore-control list-agent-runtimes \
        --query "agentRuntimes[?agentRuntimeName=='${PREFIX//-/_}_agent'].agentRuntimeId" \
        --output text | head -1)" --query 'environmentVariables' --output json 2>/dev/null)" \
    uv run python - <<'PYEOF'
import json, os
e = os.environ
env = json.loads(e["SRC"] or "{}")
env["SERVER_MODE"] = "a2a"
# It forwards to the agent Runtime rather than running a turn: a vaulted consent belongs to
# the Runtime that obtained it, so only that one can act for a user.
env["AGENT_RUNTIME_ARN"] = e["AGENT_ARN"]
# Only when .env sets it; otherwise the container's own default names the card.
if e.get("A2A_AGENT_NAME"):
    env["A2A_AGENT_NAME"] = e["A2A_AGENT_NAME"]
print(json.dumps({
    "agentRuntimeName": e["RNAME"],
    "agentRuntimeArtifact": {"containerConfiguration": {"containerUri": e["IMAGE"]}},
    "roleArn": e["ROLE"],
    "networkConfiguration": {"networkMode": "PUBLIC"},
    "protocolConfiguration": {"serverProtocol": "A2A"},
    # Authorization is stripped at the edge unless listed, and forwarding the caller's own
    # bearer is the whole design — this server adds no identity.
    "requestHeaderConfiguration": {"requestHeaderAllowlist": ["Authorization"]},
    "environmentVariables": {k: v for k, v in env.items() if v},
    "authorizerConfiguration": {"customJWTAuthorizer": {
        "discoveryUrl": e["ISSUER"] + "/.well-known/openid-configuration",
        "allowedClients": [c for c in (e["CLIENT"], e.get("WEBCLIENT", "")) if c]}},
}))
PYEOF
)"

  rid="$(aws bedrock-agentcore-control list-agent-runtimes \
    --query "agentRuntimes[?agentRuntimeName=='$rname'].agentRuntimeId" --output text 2>/dev/null | head -1)"
  if [ -n "$rid" ] && [ "$rid" != "None" ]; then
    acp update_agent_runtime "$(RID="$rid" P="$params" uv run python -c '
import json, os
p = json.loads(os.environ["P"]); p.pop("agentRuntimeName", None)
p["agentRuntimeId"] = os.environ["RID"]
print(json.dumps(p))')" >/dev/null
    echo "  updated $rname ($rid)"
  else
    rid="$(acp create_agent_runtime "$params" | uv run python -c 'import json,sys;print(json.load(sys.stdin)["agentRuntimeId"])')"
    echo "  created $rname ($rid)"
  fi
  for _ in $(seq 1 40); do
    [ "$(aws bedrock-agentcore-control get-agent-runtime --agent-runtime-id "$rid" \
         --query status --output text 2>/dev/null)" = "READY" ] && break
    sleep 5
  done
  ctx_set a2a_runtime_id "$rid"
  log "Agent card: .../runtimes/<arn>/invocations/.well-known/agent-card.json"
}


# An admin the deployment can be signed in to at once — self sign-up is off — so nobody has
# to set up Cognito before trying the agent: admin@agentcore.local, with a generated password
# kept in Secrets Manager (<name>/web/admin) and printed at the end of the deploy and by
# ./deploy.sh urls. A .local address receives no mail, so the user is created without an
# invitation and with a permanent password. WEB_ADMIN_EMAIL, when set, is invited as a second
# admin, by email with a temporary password.
WEB_ADMIN_USER="admin@agentcore.local"

web_admin() {
  local pool secret pw
  pool="$(cfn_out "$PREFIX-security" UserPoolId)"
  secret="$PREFIX/web/admin"
  pw="$(aws secretsmanager get-secret-value --secret-id "$secret" --query SecretString \
        --output text 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["password"])' 2>/dev/null || true)"
  if [ -z "$pw" ]; then
    # The pool asks for 8+ characters with lowercase letters and digits.
    pw="ac$(openssl rand -hex 8)7"
    if aws secretsmanager describe-secret --secret-id "$secret" >/dev/null 2>&1; then
      aws secretsmanager put-secret-value --secret-id "$secret" \
        --secret-string "{\"username\":\"$WEB_ADMIN_USER\",\"password\":\"$pw\"}" >/dev/null
    else
      aws secretsmanager create-secret --name "$secret" --description "Web page admin sign-in" \
        --secret-string "{\"username\":\"$WEB_ADMIN_USER\",\"password\":\"$pw\"}" >/dev/null
    fi
  fi
  if ! aws cognito-idp admin-get-user --user-pool-id "$pool" --username "$WEB_ADMIN_USER" >/dev/null 2>&1; then
    aws cognito-idp admin-create-user --user-pool-id "$pool" --username "$WEB_ADMIN_USER" \
      --message-action SUPPRESS \
      --user-attributes Name=email,Value="$WEB_ADMIN_USER" Name=email_verified,Value=true >/dev/null
  fi
  # Set every run, so the password in the secret is always the one that signs in.
  aws cognito-idp admin-set-user-password --user-pool-id "$pool" --username "$WEB_ADMIN_USER" \
    --password "$pw" --permanent
  aws cognito-idp admin-add-user-to-group --user-pool-id "$pool" --username "$WEB_ADMIN_USER" \
    --group-name admins
  echo "  admin: $WEB_ADMIN_USER (password in Secrets Manager: $secret)"

  [ -n "${WEB_ADMIN_EMAIL:-}" ] || return 0
  if aws cognito-idp admin-get-user --user-pool-id "$pool" --username "$WEB_ADMIN_EMAIL" >/dev/null 2>&1; then
    echo "  web user $WEB_ADMIN_EMAIL exists"
  else
    aws cognito-idp admin-create-user --user-pool-id "$pool" --username "$WEB_ADMIN_EMAIL" \
      --desired-delivery-mediums EMAIL \
      --user-attributes Name=email,Value="$WEB_ADMIN_EMAIL" Name=email_verified,Value=true >/dev/null
    echo "  invited $WEB_ADMIN_EMAIL: a temporary password is on its way by email"
  fi
  aws cognito-idp admin-add-user-to-group --user-pool-id "$pool" --username "$WEB_ADMIN_EMAIL" \
    --group-name admins
}

phase_webui() {
  log "WebUI — static web chat on S3 + CloudFront"
  if [ "${WEBUI:-false}" != "true" ]; then
    echo "  webui: off (set WEBUI=true in .env)"; return 0
  fi
  # The hosted login's client and domain come from the security stack, through state so the
  # page's stack holds no reference to it.
  ctx_set web_client_id "$(cfn_out "$PREFIX-security" WebClientId)"
  ctx_set web_login_domain "$(cfn_out "$PREFIX-security" WebLoginDomain)"
  $CDK deploy "$PREFIX-webui" "${CDK_CTX[@]}" --require-approval never
  local url
  url="$(cfn_out "$PREFIX-webui" WebUiUrl)"
  # CloudFront only assigns the domain now, and three stacks need it: the hosted login's
  # callback (security), CORS (router), and where a web user's consent goes back (shim).
  # It travels through .cdk-state.<name>.json, the same route the file system id takes.
  ctx_set web_origin "$url"
  $CDK deploy "$PREFIX-security" "$PREFIX-shim" "$PREFIX-router" "${CDK_CTX[@]}" \
             --exclusively --require-approval never
  log "Web chat: $url"
  web_admin
  if [ -n "${LARK_APP_ID:-}" ]; then
    echo "  inside Lark: register this domain on the Lark h5 app (docs/console-setup.md)"
  fi
  echo "  add a web user: scripts/manage-users.sh add <email>"
}


phase3_gateway() {
  # The Web Search connector is only offered in us-east-1, while everything else
  # here runs in $REGION — so this gateway lives there and the agent calls it
  # cross-region. That is fine: web search needs no end-user identity, so
  # it uses GATEWAY_IAM_ROLE outbound auth and never touches the per-user 3LO path
  # that forced the Lark tools off the Gateway in the first place.
  if [ "${WEB_SEARCH:-false}" != "true" ]; then
    log "Gateway — skipped (WEB_SEARCH is not true)"
    return 0
  fi
  # Not configurable: us-east-1 is the only region offering the connector.
  local ws_region="us-east-1"
  log "Gateway — Web Search connector in $ws_region"

  local grole gid
  grole="$(cfn_out "$PREFIX-gateway" GatewayRoleArn)"
  [ -n "$grole" ] && [ "$grole" != "None" ] || { echo "gateway stack outputs missing — run --base first"; exit 1; }

  # Inbound is IAM: the agent signs with its own role. Search reads no user data, so a
  # per-user token bought nothing and cost the agent Cognito admin rights (agent/tools/web_search.py).
  # IAM roles are global, so the role from the main region is reused as-is.
  local gw_params
  gw_params="$(ACP_NAME="${PREFIX}-websearch-gw" ACP_ROLE="$grole" uv run python -c '
import json, os
e = os.environ
print(json.dumps({
    "name": e["ACP_NAME"],
    "protocolType": "MCP",
    "protocolConfiguration": {"mcp": {
        # 2026-07-28 first, the handshake kept for older clients. The gateway serves only
        # the versions listed; asked for another, it answers 400 / -32022.
        "supportedVersions": ["2026-07-28", "2025-11-25"],
        # Without this the gateway issues no Mcp-Session-Id and every tools/call
        # cold-starts a fresh downstream microVM (measured; see docs).
        "sessionConfiguration": {"sessionTimeoutInSeconds": 3600}}},
    "roleArn": e["ACP_ROLE"],
    "authorizerType": "AWS_IAM",
}))')"
  gid="$(aws bedrock-agentcore-control list-gateways --region "$ws_region" \
    --query "items[?name=='${PREFIX}-websearch-gw'].gatewayId" --output text 2>/dev/null || true)"
  # A gateway created when inbound was CUSTOM_JWT. Its authorizer type cannot be updated
  # (measured: ValidationException), so it is replaced. It holds no state — a target and
  # a URL, and the runtime step after this one picks up the new URL.
  if [ -n "$gid" ] && [ "$gid" != "None" ] && [ "$(aws bedrock-agentcore-control get-gateway \
       --region "$ws_region" --gateway-identifier "$gid" --query authorizerType --output text)" != "AWS_IAM" ]; then
    echo "  replacing gateway $gid: its inbound auth predates AWS_IAM"
    local old_tid
    for old_tid in $(aws bedrock-agentcore-control list-gateway-targets --region "$ws_region" \
                       --gateway-identifier "$gid" --query 'items[].targetId' --output text); do
      aws bedrock-agentcore-control delete-gateway-target --region "$ws_region" \
        --gateway-identifier "$gid" --target-id "$old_tid" >/dev/null
    done
    for _ in $(seq 1 40); do
      [ -z "$(aws bedrock-agentcore-control list-gateway-targets --region "$ws_region" \
              --gateway-identifier "$gid" --query 'items[].targetId' --output text)" ] && break
      sleep 3
    done
    aws bedrock-agentcore-control delete-gateway --region "$ws_region" --gateway-identifier "$gid" >/dev/null
    for _ in $(seq 1 40); do
      aws bedrock-agentcore-control get-gateway --region "$ws_region" --gateway-identifier "$gid" \
        >/dev/null 2>&1 || break
      sleep 3
    done
    gid=""
  fi
  if [ -z "$gid" ] || [ "$gid" = "None" ]; then
    echo "  creating gateway"
    gid="$(acp create_gateway "$gw_params" "$ws_region" | uv run python -c 'import json,sys;print(json.load(sys.stdin)["gatewayId"])')"
  elif [ "$(aws bedrock-agentcore-control get-gateway --region "$ws_region" --gateway-identifier "$gid" \
            --query "join(',', protocolConfiguration.mcp.supportedVersions)" --output text)" != "2026-07-28,2025-11-25" ]; then
    # A gateway created with the handshake only. An account not enabled for MCP version
    # updates refuses this; the agent then reaches the gateway through the handshake anyway.
    echo "  updating gateway $gid to serve MCP 2026-07-28"
    acp update_gateway "$(GW_PARAMS="$gw_params" GID="$gid" uv run python -c 'import json,os
print(json.dumps({**json.loads(os.environ["GW_PARAMS"]), "gatewayIdentifier": os.environ["GID"]}))')" \
      "$ws_region" >/dev/null || echo "  (version update refused; the gateway keeps the handshake)"
  fi
  # A gateway is briefly CREATING/UPDATING; targets can't be added until it settles.
  for _ in $(seq 1 30); do
    [ "$(aws bedrock-agentcore-control get-gateway --region "$ws_region" \
         --gateway-identifier "$gid" --query status --output text 2>/dev/null)" = "READY" ] && break
    sleep 3
  done
  echo "  gateway: $gid"

  # The tool name must be WebSearch and parameterValues must be present even when
  # empty ({} = no domain filter); the API rejects a config entry without it.
  local tid
  tid="$(aws bedrock-agentcore-control list-gateway-targets --region "$ws_region" \
    --gateway-identifier "$gid" --query "items[?name=='web-search-tool'].targetId" \
    --output text 2>/dev/null || true)"
  if [ -z "$tid" ] || [ "$tid" = "None" ]; then
    echo "  creating web-search target"
    tid="$(acp create_gateway_target "$(ACP_GID="$gid" uv run python -c '
import json, os
print(json.dumps({
    "gatewayIdentifier": os.environ["ACP_GID"],
    "name": "web-search-tool",
    # The tool name must be WebSearch, and parameterValues has to be present even when
    # empty ({} = no domain filter) — the API rejects a config entry without it.
    "targetConfiguration": {"mcp": {"connector": {
        "source": {"connectorId": "web-search"},
        "configurations": [{"name": "WebSearch", "parameterValues": {}}]}}},
    "credentialProviderConfigurations": [{"credentialProviderType": "GATEWAY_IAM_ROLE"}],
}))')" "$ws_region" | uv run python -c 'import json,sys;print(json.load(sys.stdin)["targetId"])')"
  fi
  echo "  target: $tid"

  local gurl
  gurl="$(aws bedrock-agentcore-control get-gateway --region "$ws_region" \
    --gateway-identifier "$gid" --query gatewayUrl --output text 2>/dev/null || true)"
  ctx_set websearch_gateway_id "$gid"
  [ -n "$gurl" ] && [ "$gurl" != "None" ] && ctx_set websearch_gateway_url "$gurl"
  echo "  url: $gurl"
  echo "  next: ./deploy.sh runtime  (injects the URL into the agent)"
}

case "${1:-all}" in
  --base|--phase1) base_cdk_stacks ;;  # --phase1 kept as a back-compat alias
  --memory)   phase_memory ;;
  # Memory first: the runtime is created with BEDROCK_AGENTCORE_MEMORY_ID baked into its
  # environment, and UpdateAgentRuntime replaces rather than patches. The A2A runtime runs the
  # same image with the agent's env, so it follows; left behind, it pointed at an image tag the
  # repository had since pruned and failed every call.
  --runtime)  phase_memory; phase_code; phase2_runtime; phase_a2a ;;
  --code)     phase_code ;;
  --a2a)      phase_a2a ;;
  --webui)    phase_webui ;;
  --gateway)  phase3_gateway ;;
  all|"")     base_cdk_stacks; phase_memory; phase_code; phase2_runtime; phase3_gateway; phase_a2a; phase_webui
              [ "$LARK" != "true" ] || log "Webhook URL (register in Lark): $(cfn_out "$PREFIX-router" WebhookLarkUrl)" ;;
  *) echo "usage: [PROFILE=p REGION=r] $0 [--base|--memory|--code|--runtime|--gateway|--a2a|--webui]"; exit 1 ;;
esac
exit
}
