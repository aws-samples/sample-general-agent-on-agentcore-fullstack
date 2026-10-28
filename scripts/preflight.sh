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
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }
ver() { grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1 || true; }   # first x.y[.z], or nothing
# A tool that cannot run (a version-manager shim with no version selected, the macOS python3
# stub without developer tools) reads as no version, not a silent exit under pipefail.
version_of() { { "$@" 2>&1 || true; } | ver; }
ver_ge() {  # $1 >= $2, numerically per field
  awk -v a="$1" -v b="$2" 'BEGIN { split(a, x, "."); split(b, y, ".")
    for (i = 1; i <= 3; i++) { if (x[i] + 0 != y[i] + 0) exit !(x[i] + 0 > y[i] + 0) } }'
}

# --- tools ---------------------------------------------------------------
missing=""
for t in uv aws node npx python3 zip openssl $([ "$AGENT_BUILD" = "local" ] && echo docker); do
  command -v "$t" >/dev/null || missing="$missing $t"
done
[ -z "$missing" ] || fail "missing tools:$missing
  uv      https://docs.astral.sh/uv/getting-started/installation/
  aws     https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
  node    https://nodejs.org/ (22 or 24 LTS; npx comes with it)
  zip     macOS has it; Ubuntu / WSL: sudo apt install zip"

# Floors are what was measured: uv 0.4 cannot read uv.lock and rewrites it, the CDK CLI
# refuses Node < 18, the system python3 only runs stdlib one-liners.
uv_v="$(version_of uv --version)"; node_v="$(version_of node --version)"
py_v="$(version_of python3 --version)"
# After aws-cli/: on an old Python the CLI may print a warning with its own x.y first.
aws_v="$({ aws --version 2>&1 || true; } | sed -nE 's|.*aws-cli/([0-9.]+).*|\1|p' | ver)"
old=""
ver_ge "$uv_v" 0.5.0   || old="$old
  uv ${uv_v:-?} < 0.5 (or not runnable) — upgrade: uv self update (or your package manager)"
ver_ge "$aws_v" 2.0.0  || old="$old
  AWS CLI ${aws_v:-?} is not v2 — install AWS CLI v2"
ver_ge "$node_v" 18.0.0 || old="$old
  Node.js ${node_v:-?} < 18 (or not runnable) — install Node.js 22 or 24 LTS"
ver_ge "$py_v" 3.8.0   || old="$old
  python3 ${py_v:-?} < 3.8 (or not runnable)"
[ -z "$old" ] || fail "tools too old:$old"
# The CDK's Python library runs on Node through jsii, which supports 22 and 24 and prints a
# banner on anything else (20 deprecated, 26 untested).
case "${node_v%%.*}" in 22|24) ;; *)
  warn "  warning: Node.js $node_v is not one the CDK supports (22, 24) — expect warnings; install 22 or 24 LTS if the deploy misbehaves" ;;
esac
echo "  tools   : uv $uv_v · aws-cli $aws_v · node $node_v · python3 $py_v"

# Images build in CodeBuild unless AGENT_BUILD=local, which builds the agent with this
# host's Docker — then a stopped daemon would fail the deploy minutes in rather than here.
if [ "$AGENT_BUILD" = "local" ]; then
  docker info >/dev/null 2>&1 || fail "Docker isn't running — AGENT_BUILD=local builds the agent image with it."
fi

# --- downloads -----------------------------------------------------------
# Every later step runs through `uv run` and `npx aws-cdk@2`, and the first of each in a fresh
# clone downloads quietly: Python 3.13+ and ~250 MB of packages, then the CDK CLI. Done here,
# with progress shown, so a slow network reads as a download rather than a hang.
if ! uv sync --offline --quiet 2>/dev/null; then
  echo "  python  : first run — downloading Python 3.13+ and the project's packages (~300 MB)…"
  # Plain sync, not --locked: with a mirror index the lock's pypi.org URLs read as stale.
  uv sync || fail "uv could not set up the Python environment. On a slow or restricted network, use mirrors and re-run:
  export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
  export UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone/"
fi
echo "  python  : $(uv run python --version 2>&1) (project environment)"
printf '  cdk     : '
cdk_v="$(npx --yes aws-cdk@2 --version 2>&1 | ver)" && [ -n "$cdk_v" ] \
  || { echo; fail "could not fetch the CDK CLI from npm. On a slow or restricted network, use a mirror and re-run:
  export npm_config_registry=https://registry.npmmirror.com"; }
echo "aws-cdk $cdk_v"

# --- credentials ---------------------------------------------------------
echo "  checking AWS credentials and permissions…"
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
denied="" unknown=""
probe() {  # label, command…
  local label="$1" err; shift
  err="$("$@" 2>&1 >/dev/null)" && return
  # A CLI that predates the service rejects the command locally — not a denial.
  case "$err" in *"nvalid choice"*) unknown="$unknown $label" ;; *) denied="$denied $label" ;; esac
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
[ "${BROWSER:-false}" != "true" ] || probe browser \
  aws bedrock-agentcore list-browser-sessions --browser-identifier aws.browser.v1 --max-results 1
if [ "${FILES_STORAGE:-false}" = "true" ]; then
  # boto3, not `aws s3files`: that subcommand is absent from AWS CLI 2.34, so probing it
  # through the CLI reports a denial where the real problem is the CLI's vocabulary.
  probe s3files uv run --with boto3 python -c \
    "import boto3; boto3.client('s3files').list_file_systems(maxResults=1)"
  probe kms     aws kms list-keys --limit 1
  probe ec2     aws ec2 describe-vpcs --max-items 1
fi
[ -z "$unknown" ] || fail "AWS CLI $aws_v has no command for:$unknown — upgrade to the latest AWS CLI v2"
[ -z "$denied" ] || printf '\033[1;33m%s\033[0m\n' "  warning: no read access to:$denied
  The deploy writes to these — expect it to fail unless that is just a read-only denial."

# --- network -------------------------------------------------------------
# The agent Runtime's network. A mismatch with what the deployment was created with, an
# unsupported AZ or a subnet that is not private fails at Runtime creation, minutes in.
. scripts/lib/network.sh
network_check_lock
if [ "$AGENT_NETWORK" = "vpc" ]; then
  [ -n "$(supported_az_ids "$REGION")" ] || fail "AgentCore lists no VPC-capable AZs for $REGION — use AGENT_NETWORK=public"
  if [ "$NETWORK_KIND" = "vpc" ]; then
    [ "$(supported_az_names | tr ',' '\n' | grep -c .)" -ge 2 ] \
      || fail "this account has fewer than two AgentCore-supported AZs in $REGION ($(supported_az_ids "$REGION"))"
  else
    IFS=, read -r -a subnet_ids <<< "$AGENT_SUBNET_IDS"
    subnets_json="$(aws ec2 describe-subnets --subnet-ids "${subnet_ids[@]}" --output json 2>&1)" \
      || fail "cannot read AGENT_SUBNET_IDS: $subnets_json"
    vpc_id="$(printf '%s' "$subnets_json" | python3 -c 'import json,sys; print(" ".join(sorted({s["VpcId"] for s in json.load(sys.stdin)["Subnets"]})))')"
    [ "$(echo "$vpc_id" | wc -w)" -eq 1 ] || fail "AGENT_SUBNET_IDS span more than one VPC ($vpc_id)"
    # Private: no route to an internet gateway, whether the table is associated or the main one.
    routes_json="$(aws ec2 describe-route-tables --filters "Name=vpc-id,Values=$vpc_id" --output json 2>&1)" \
      || fail "cannot read the route tables of $vpc_id: $routes_json"
    endpoints_json="$(aws ec2 describe-vpc-endpoints --filters "Name=vpc-id,Values=$vpc_id" \
      "Name=service-name,Values=com.amazonaws.$REGION.s3" "Name=vpc-endpoint-type,Values=Gateway" \
      --output json 2>&1)" || fail "cannot read the VPC endpoints of $vpc_id: $endpoints_json"
    dns="$(for a in enableDnsSupport enableDnsHostnames; do
      aws ec2 describe-vpc-attribute --vpc-id "$vpc_id" --attribute "$a" --query "*.Value | [0]" --output text 2>/dev/null \
        || echo unreadable; done | tr '\n' ' ')"
    problems="$(SUBNETS="$subnets_json" ROUTES="$routes_json" OK_IDS="$(supported_az_ids "$REGION")" \
      ENDPOINTS="$endpoints_json" DNS="$dns" python3 - <<'PYEOF'
import json, os
subnets = json.loads(os.environ["SUBNETS"])["Subnets"]
tables = json.loads(os.environ["ROUTES"])["RouteTables"]
ok = set(os.environ["OK_IDS"].split())
s3_tables = {t for e in json.loads(os.environ["ENDPOINTS"])["VpcEndpoints"] for t in e.get("RouteTableIds", [])}
main = next((t for t in tables if any(a.get("Main") for a in t.get("Associations", []))), None)
out = []
for s in subnets:
    if s["AvailabilityZoneId"] not in ok:
        out.append(f'{s["SubnetId"]} is in {s["AvailabilityZoneId"]}, not one AgentCore supports ({" ".join(sorted(ok))})')
    # A subnet without its own association uses the main table.
    t = next((t for t in tables if any(a.get("SubnetId") == s["SubnetId"] for a in t.get("Associations", []))), main)
    if not t:
        continue
    if any(r.get("GatewayId", "").startswith("igw-") for r in t.get("Routes", [])):
        out.append(f'{s["SubnetId"]} routes to an internet gateway; AgentCore needs private subnets')
    if t["RouteTableId"] not in s3_tables:
        out.append(f'{s["SubnetId"]}: no S3 gateway endpoint on its route table {t["RouteTableId"]}; a new Runtime pulls its image layers through one')
    if not any(r.get("DestinationCidrBlock") == "0.0.0.0/0" for r in t.get("Routes", [])):
        print(f'WARN {s["SubnetId"]} has no default route: Lark, Google and the us-east-1 Gateway will be unreachable unless a proxy or endpoint serves them')
if len({s["AvailabilityZoneId"] for s in subnets if s["AvailabilityZoneId"] in ok}) < 2:
    out.append("AGENT_SUBNET_IDS need at least two supported AZs")
if "False" in os.environ["DNS"] or "unreadable" in os.environ["DNS"]:
    out.append("the VPC needs DNS support and DNS hostnames enabled (or they could not be read)")
print("\n".join(f"  {p}" for p in out))
PYEOF
)"
    while IFS= read -r line; do warn "  warning:${line#WARN}"; done < <(printf '%s\n' "$problems" | grep '^WARN' || true)
    problems="$(printf '%s\n' "$problems" | grep -v '^WARN' || true)"
    [ -z "$problems" ] || fail "AGENT_SUBNET_IDS cannot host the agent:
$problems"
  fi
fi

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
# A GitHub app is both keys or neither: one alone creates no provider and says nothing.
if { [ -n "${GITHUB_CLIENT_ID:-}" ] && [ -z "${GITHUB_CLIENT_SECRET:-}" ]; } || \
   { [ -z "${GITHUB_CLIENT_ID:-}" ] && [ -n "${GITHUB_CLIENT_SECRET:-}" ]; }; then
  fail "GitHub is half configured — set both GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET in .env, or neither"
fi
if [ "$LARK" != "true" ]; then
  # Without Lark the web page is the only way in, and someone has to be able to sign in.
  [ "${WEBUI:-false}" = "true" ] || fail "no Lark app and WEBUI is not true — nobody could reach the agent"
  [ -z "${AGENT_DECIDE_APPROVAL_CODES:-}" ] || fail "AGENT_DECIDE_APPROVAL_CODES needs Lark (approvals are Lark's)"
fi

echo "  deploy  : $DEPLOYMENT_NAME   (state: $STATE_FILE)"
echo "  account : $ACCOUNT"
echo "  region  : $REGION${PROFILE:+   profile: $PROFILE}"
echo "  model   : $MODEL_ID"
echo "  lark    : $([ "$LARK" = "true" ] && echo "$LARK_API_DOMAIN" || echo 'off  (web page only)')"
echo "  build   : $AGENT_BUILD$([ "$AGENT_BUILD" = "local" ] && echo '  (local Docker)' || echo '  (CodeBuild, ARM64 native)')"
case "$NETWORK_KIND" in
  public)  echo "  network : agent PUBLIC" ;;
  vpc:byo) echo "  network : agent in VPC $vpc_id (your subnets; your network decides its egress)" ;;
  vpc)     echo "  network : agent in a created VPC (one NAT with a fixed IP, a fixed hourly cost; flow logs)" ;;
esac
echo "  search  : $WEB_SEARCH$([ "$WEB_SEARCH" = "true" ] && echo '  (+ a gateway in us-east-1)')"
echo "  browser : ${BROWSER:-false}$([ "${BROWSER:-false}" = "true" ] && echo '  (AgentCore Browser, per second while one runs)' || true)"
echo "  files   : ${FILES_STORAGE:-false}$([ "${FILES_STORAGE:-false}" = "true" ] && { [ "$NETWORK_KIND" = "vpc" ] && echo "  (S3 Files, in the agent's VPC)" || echo "  (creates a VPC + S3 Files)"; } || true)"
if [ "${FILES_STORAGE:-false}" = "true" ]; then
  echo "  nat     : ${NAT_GATEWAY:-false}$([ "${NAT_GATEWAY:-false}" = "true" ] && echo '  (code sandbox gets internet; NAT is a fixed hourly cost)' || true)"
fi
