#!/usr/bin/env bash
# Tear down everything deploy.sh created, in dependency order. Override the
# target with PROFILE=... REGION=... env vars (same as deploy.sh).
#
# The AgentCore Runtimes are created out-of-band by the control-plane CLI, NOT by
# CloudFormation — so `cdk destroy` alone can't remove them. This script deletes
# them first, then destroys the CDK stacks.
#
# Order: what the CLI created (runtimes, code interpreters, gateway, Memory, images) →
# S3 Files access points → every CDK stack → identity and secrets → log groups last.
# The access points belong to no stack yet pin the storage stack's file system. AgentCore's
# network interfaces in its subnets pin it too, until AgentCore releases them (up to 8 h);
# while they remain, that stack and the two it imports from are left for a re-run.
# Everything is discovered dynamically by name (no hardcoded ids), so this is
# safe to re-run — already-gone resources are skipped.
#
# Usage: [PROFILE=p REGION=r] scripts/destroy.sh [--yes]
#   --yes   skip the interactive confirmation
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Deployment target: command-line env vars win over .env, which wins over defaults.
_CLI_PROFILE="${PROFILE:-}" _CLI_REGION="${REGION:-}"
[ -f .env ] && { set -a; . ./.env; set +a; }
PROFILE="${_CLI_PROFILE:-${PROFILE:-}}"   # empty -> ambient creds (instance role / env)
REGION="${_CLI_REGION:-${REGION:-us-west-2}}"
PREFIX="agentcore-fullstack"
# boto3 reads AWS_DEFAULT_REGION or the profile's region, never AWS_REGION.
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"
# Credentials already in the environment outrank .env's profile.
[ -n "${AWS_ACCESS_KEY_ID:-}" ] || { [ -n "$PROFILE" ] && export AWS_PROFILE="$PROFILE"; } || true

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
export CDK_DEFAULT_ACCOUNT="$ACCOUNT" CDK_DEFAULT_REGION="$REGION"
CDK="npx --yes aws-cdk@2"
# AgentCore resource names use underscores.
U="${PREFIX//-/_}"
# The Web Search connector exists only in us-east-1, so its gateway does too.
WS_REGION="us-east-1"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }

# --- confirmation --------------------------------------------------------
if [ "${1:-}" != "--yes" ]; then
  warn "This will DELETE all $PREFIX resources in account $ACCOUNT / $REGION:"
  warn "  every ${U}_* Runtime and Code Interpreter, the Memory (all long-term records),"
  warn "  the Web Search gateway in $WS_REGION, and every CDK stack of this app."
  warn "  Conversation state goes with the agentcore stack."
  warn "  Image builds go too: the $PREFIX-* ECR repos and their images, the CodeBuild"
  warn "  image builder and its role $PREFIX-image-builder-$REGION."
  warn "  With files storage deployed: every user's access point goes too (their files stay"
  warn "  in the bucket until the bucket itself is removed), along with the VPC."
  warn "  Secrets include your Lark credentials ($PREFIX/channels/lark) — re-seedable"
  warn "  from .env via scripts/setup-lark.sh. Lark console app config is NOT touched."
  warn "  The OAuth provider goes too, so every user's vaulted token is purged and the"
  warn "  next deploy issues a NEW callbackUrl to register in the Lark console."
  read -r -p "Type the account id ($ACCOUNT) to proceed: " reply
  [ "$reply" = "$ACCOUNT" ] || { echo "aborted."; exit 1; }
fi


# --- 1. runtimes, code interpreters, gateway (CLI-created) ---------------
gone() { # seconds, then a command that succeeds while the thing still exists
  local t="$1"; shift
  for _ in $(seq 1 $((t / 5))); do "$@" >/dev/null 2>&1 || return 0; sleep 5; done
  return 1
}

log "Runtimes — every ${U}_* AgentCore Runtime"
for rid in $(aws bedrock-agentcore-control list-agent-runtimes \
               --query "agentRuntimes[?starts_with(agentRuntimeName,'${U}_')].agentRuntimeId" \
               --output text 2>/dev/null || true); do
  echo "  deleting runtime $rid"
  aws bedrock-agentcore-control delete-agent-runtime --agent-runtime-id "$rid" >/dev/null 2>&1 \
    || warn "  (runtime $rid delete failed)"
done

# Deletion is requested and not waited for. The network interfaces an interpreter leaves in
# the sandbox subnets belong to AgentCore, which releases them on its own schedule — up to
# 8 hours, per its VPC docs — and nothing in this account can detach them. Step 2 deals with
# what that holds up.
log "Code Interpreters — every ${U}_* interpreter"
for cid in $(aws bedrock-agentcore-control list-code-interpreters \
               --query "codeInterpreterSummaries[?starts_with(name,'${U}_') && status!='DELETED'].codeInterpreterId" \
               --output text 2>/dev/null || true); do
  echo "  deleting code interpreter $cid"
  aws bedrock-agentcore-control delete-code-interpreter --code-interpreter-id "$cid" >/dev/null 2>&1 \
    || warn "  (code interpreter $cid delete failed)"
done

log "Gateway — Web Search in $WS_REGION"
for gid in $(aws bedrock-agentcore-control list-gateways --region "$WS_REGION" \
               --query "items[?starts_with(name,'$PREFIX')].gatewayId" --output text 2>/dev/null || true); do
  # A gateway with targets cannot be deleted.
  for tid in $(aws bedrock-agentcore-control list-gateway-targets --region "$WS_REGION" \
                 --gateway-identifier "$gid" --query 'items[].targetId' --output text 2>/dev/null || true); do
    aws bedrock-agentcore-control delete-gateway-target --region "$WS_REGION" \
      --gateway-identifier "$gid" --target-id "$tid" >/dev/null 2>&1 || true
  done
  target_left() {
    [ -n "$(aws bedrock-agentcore-control list-gateway-targets --region "$WS_REGION" \
              --gateway-identifier "$1" --query 'items[].targetId' --output text 2>/dev/null)" ]
  }
  gone 120 target_left "$gid" || true
  aws bedrock-agentcore-control delete-gateway --region "$WS_REGION" --gateway-identifier "$gid" >/dev/null 2>&1 \
    && echo "  deleted gateway $gid" || warn "  (gateway $gid delete failed)"
done

# --- 1b. Memory, ECR, CodeBuild (created by the deploy scripts) -----------
# None of these belong to a stack, and leaving them behind is not harmless: the
# Memory holds every user's conversation history and keeps billing for stored
# events, and the ECR repo keeps paying for image storage. A redeploy creates a
# new Memory rather than adopting this one, so it would just accumulate.
log "Memory / ECR / CodeBuild — CLI-created leftovers"
MEM_PREFIX="${PREFIX//-/_}"
for mid in $(aws bedrock-agentcore-control list-memories --max-results 100 \
               --query "memories[?starts_with(id,'$MEM_PREFIX')].id" --output text 2>/dev/null || true); do
  echo "  deleting memory $mid (conversation history)"
  aws bedrock-agentcore-control delete-memory --memory-id "$mid" >/dev/null 2>&1 \
    || warn "  (memory $mid delete failed)"
done

# Two naming schemes: our own $PREFIX-*, and the AgentCore CLI's
# bedrock-agentcore-<runtime_name>* (repo and builder project alike).
for repo in $(aws ecr describe-repositories \
                --query "repositories[?starts_with(repositoryName,'$PREFIX') || starts_with(repositoryName,'bedrock-agentcore-${PREFIX//-/_}')].repositoryName" \
                --output text 2>/dev/null || true); do
  echo "  deleting ECR repo $repo (with images)"
  aws ecr delete-repository --repository-name "$repo" --force >/dev/null 2>&1 \
    || warn "  (ECR repo $repo delete failed)"
done

for proj in $(aws codebuild list-projects --query "projects[?starts_with(@,'$PREFIX') || starts_with(@,'bedrock-agentcore-${PREFIX//-/_}')]" \
                --output text 2>/dev/null || true); do
  echo "  deleting CodeBuild project $proj"
  aws codebuild delete-project --name "$proj" >/dev/null 2>&1 \
    || warn "  (CodeBuild project $proj delete failed)"
done

# The image builder's service role (scripts/lib/codebuild.sh). Not a CDK resource, and
# named per region, so only this region's is removed.
CB_ROLE_NAME="$PREFIX-image-builder-$REGION"
if aws iam get-role --role-name "$CB_ROLE_NAME" >/dev/null 2>&1; then
  for pol in $(aws iam list-role-policies --role-name "$CB_ROLE_NAME" --query 'PolicyNames' --output text); do
    aws iam delete-role-policy --role-name "$CB_ROLE_NAME" --policy-name "$pol" >/dev/null 2>&1 || true
  done
  aws iam delete-role --role-name "$CB_ROLE_NAME" >/dev/null 2>&1 \
    && echo "  deleted IAM role $CB_ROLE_NAME" || warn "  (IAM role $CB_ROLE_NAME delete failed)"
fi

# The CodeBuild source bucket is shared by every AgentCore project in the account
# (its name carries no project prefix), so delete only our own source keys and
# leave the bucket alone.
SRC_BUCKET="bedrock-agentcore-codebuild-sources-$ACCOUNT-$REGION"
if aws s3api head-bucket --bucket "$SRC_BUCKET" >/dev/null 2>&1; then
  # One key per image, <repo>/source.zip, every repo named $PREFIX-*.
  aws s3 rm "s3://$SRC_BUCKET/" --recursive --exclude '*' --include "$PREFIX-*/source.zip" 2>/dev/null \
    | sed 's/^delete: /  removed /' || true
fi

# --- 1c. S3 Files Access Points (created at runtime by the broker) --------
# One per user, created on first mount — so CloudFormation has never heard of them, and
# deleting the file system while they exist either fails or orphans them. They also keep
# the file system undeletable, which then blocks the whole storage stack.
#
# Done through boto3 rather than `aws s3files`: that subcommand is missing from AWS CLI
# 2.34 (verified), so a CLI-based delete fails silently on an otherwise current machine.
# The service itself is fine — the API answers, the CLI just has no word for it yet.
if aws cloudformation describe-stacks --stack-name "$PREFIX-storage" >/dev/null 2>&1; then
  log "S3 Files — per-user access points"
  fs_id="$(aws cloudformation describe-stacks --stack-name "$PREFIX-storage" \
    --query "Stacks[0].Outputs[?OutputKey=='FileSystemId'].OutputValue" \
    --output text 2>/dev/null)"
  if [ -n "$fs_id" ] && [ "$fs_id" != "None" ]; then
    FS_ID="$fs_id" uv run --with boto3 python - <<'PYEOF' || warn "  (access point cleanup failed)"
import os, boto3
fs = os.environ["FS_ID"]
c = boto3.client("s3files")
n, token = 0, None
while True:
    kw = {"fileSystemId": fs, "maxResults": 100}
    if token:
        kw["nextToken"] = token
    page = c.list_access_points(**kw)
    for ap in page.get("accessPoints", []):
        ap_id = ap["accessPointId"]
        try:
            c.delete_access_point(accessPointId=ap_id)
            print(f"  deleted access point {ap_id} ({(ap.get('rootDirectory') or {}).get('path','?')})")
            n += 1
        except Exception as e:
            print(f"  access point {ap_id} delete failed: {type(e).__name__}")
    token = page.get("nextToken")
    if not token:
        break
print(f"  {n} access point(s) removed; the files themselves stay in the bucket")
PYEOF
  fi
fi

# --- 2. CDK stacks --------------------------------------------------------
# Every optional stack switched on, so whatever was deployed is in the app; a stack that was
# never deployed is skipped, and CDK orders the deletions by dependency.
CDK_CTX=(-c files_storage=true -c webui=true)
# While AgentCore still holds interfaces in the sandbox subnets, the storage stack cannot
# delete them (DependencyViolation), and agentcore and security, which it imports from,
# cannot go before it. Trying anyway leaves all three DELETE_FAILED; leaving them for a
# re-run does not.
vpc="$(aws ec2 describe-vpcs --filters "Name=tag:aws:cloudformation:stack-name,Values=$PREFIX-storage" \
         --query 'Vpcs[0].VpcId' --output text 2>/dev/null || true)"
held=""
if [ -n "$vpc" ] && [ "$vpc" != "None" ]; then
  held="$(aws ec2 describe-network-interfaces --filters "Name=vpc-id,Values=$vpc" \
            "Name=interface-type,Values=agentic_ai" --query 'NetworkInterfaces[].NetworkInterfaceId' \
            --output text 2>/dev/null || true)"
fi
KEPT=("$PREFIX-storage" "$PREFIX-agentcore" "$PREFIX-security")
if [ -z "$held" ]; then
  log "CDK — destroy every stack of this app"
  $CDK destroy --all "${CDK_CTX[@]}" --force
else
  log "CDK — destroy every stack but ${KEPT[*]}"
  warn "  AgentCore has not yet released $(echo "$held" | wc -w | tr -d ' ') network interface(s) in $vpc ($held)."
  others=()
  for s in $($CDK list "${CDK_CTX[@]}" 2>/dev/null); do
    case " ${KEPT[*]} " in *" $s "*) ;; *) others+=("$s") ;; esac
  done
  [ ${#others[@]} -eq 0 ] || $CDK destroy "${others[@]}" "${CDK_CTX[@]}" --force
fi

# --- 3. AgentCore Identity (CLI-created, region-scoped) ------------------
# Deleting the provider purges every user's vaulted token, so a redeploy means
# everyone consents again — and the new provider gets a NEW callbackUrl that must
# be registered in the Lark console. Both are region-scoped: leaving them behind
# is what litters an account after moving regions.
log "AgentCore Identity — OAuth provider + workload identity"
# Every $PREFIX-* provider: Lark (-3lo) and, when configured, Google (-google).
for prov in $(aws bedrock-agentcore-control list-oauth2-credential-providers \
                --query "credentialProviders[?starts_with(name,'$PREFIX')].name" --output text 2>/dev/null || true); do
  aws bedrock-agentcore-control delete-oauth2-credential-provider --name "$prov" >/dev/null 2>&1 \
    && echo "  deleted provider $prov (vaulted tokens purged)" \
    || warn "  (provider $prov delete failed)"
done
if aws bedrock-agentcore-control get-workload-identity --name "$PREFIX-wl" >/dev/null 2>&1; then
  aws bedrock-agentcore-control delete-workload-identity --name "$PREFIX-wl" >/dev/null 2>&1 \
    && echo "  deleted workload identity $PREFIX-wl" \
    || warn "  (workload $PREFIX-wl delete failed — service-linked ones can't be deleted)"
else
  echo "  no workload identity named $PREFIX-wl — skipping"
fi

# A secret CloudFormation deleted is only scheduled for deletion, and its name stays taken
# until the recovery window ends — so a redeploy would fail creating it. Purged here, and
# only those: --include-planned-deletion lists live secrets too, and a stack kept for a
# re-run still owns its own.
log "Secrets — purge $PREFIX/* scheduled for deletion"
for sid in $(aws secretsmanager list-secrets --include-planned-deletion \
               --query "SecretList[?starts_with(Name,'$PREFIX/') && DeletedDate].ARN" --output text 2>/dev/null || true); do
  aws secretsmanager delete-secret --secret-id "$sid" --force-delete-without-recovery >/dev/null 2>&1 \
    && echo "  purged ${sid##*:secret:}" || warn "  (secret $sid purge failed)"
done

# --- 4. Log groups (auto-created by runtimes / CodeBuild / Memory) --------
# Last, because everything above can still write logs while it is being deleted.
log "Log groups — runtime / CodeBuild / Memory leftovers"
for g in $(aws logs describe-log-groups \
             --query "logGroups[?contains(logGroupName,'$PREFIX') || contains(logGroupName,'${PREFIX//-/_}')].logGroupName" \
             --output text 2>/dev/null || true); do
  aws logs delete-log-group --log-group-name "$g" >/dev/null 2>&1 \
    && echo "  deleted $g" || true
done

# --- 5. leftover local state --------------------------------------------
log "Local — clear the AgentCore CLI config so a fresh deploy reconfigures"
[ -f .bedrock_agentcore.yaml ] && { rm -f .bedrock_agentcore.yaml; echo "  removed .bedrock_agentcore.yaml"; }  # safe-rm-ok
rm -rf .bedrock_agentcore 2>/dev/null || true  # safe-rm-ok

# .cdk-state.json holds this deployment's ids; a stale one would point a fresh
# deploy at resources that no longer exist.
[ -f .cdk-state.json ] && { rm -f .cdk-state.json; echo "  removed .cdk-state.json"; }  # safe-rm-ok

if [ -n "$held" ]; then
  log "Done, except ${KEPT[*]}."
  warn "Those wait on AgentCore to release its interfaces in $vpc (up to 8 hours after the"
  warn "interpreter was deleted). Run scripts/destroy.sh again then; it picks up where this left off."
else
  log "Done. Lark console app config is untouched; re-deploy with ./deploy.sh."
fi
warn "The webhook, web chat and OAuth callback URLs all change on redeploy; ./deploy.sh"
warn "prints the ones to register in the Lark console (and the Google console, if used)."
