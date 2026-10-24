# The deployment target, set up the same way for every script. Source it first; it cds to
# the repo root.
#
#   - .env is the config; a variable already set in the environment wins over it, so
#     `REGION=us-east-1 ./deploy.sh` works and survives into every step deploy.sh runs.
#   - REGION is required and reaches every client: the AWS CLI reads AWS_REGION, boto3 reads
#     only AWS_DEFAULT_REGION (and otherwise falls back to the profile's region — measured,
#     it sent the S3 Files cleanup to another region).
#   - Credentials already in the environment outrank .env's PROFILE.
#   - DEPLOYMENT_NAME (PREFIX here) starts every resource name, and names this checkout's
#     state file and CDK output dir, so two deployments never read each other's ids.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

[ -f .env ] || { echo "missing .env — copy .env.example to .env and fill it in" >&2; exit 1; }

# Remember what the caller set (non-empty) for any key .env defines, source .env, then put
# those back.
_env_keep=""
for _k in $(sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' .env); do
  if [ -n "${!_k:-}" ]; then
    _env_keep="$_env_keep$(printf 'export %s=%q;' "$_k" "${!_k}")"
  fi
done
set -a; . ./.env; set +a
eval "$_env_keep"
unset _env_keep _k

: "${REGION:?set REGION in .env}"
: "${DEPLOYMENT_NAME:?set DEPLOYMENT_NAME in .env}"
# 23: "<name>-checkpoints-<account>-<region>" must fit S3's 63 in the longest region names.
[[ "$DEPLOYMENT_NAME" =~ ^[a-z][a-z0-9-]{2,22}$ && "$DEPLOYMENT_NAME" != *- ]] || {
  echo "DEPLOYMENT_NAME '$DEPLOYMENT_NAME' must be 3-23 chars of a-z, 0-9 and '-', starting with a letter" >&2
  exit 1; }
PREFIX="$DEPLOYMENT_NAME"
export DEPLOYMENT_NAME
STATE_FILE=".cdk-state.$PREFIX.json"   # this deployment's ids (gitignored)
CDK_OUT="cdk.out/$PREFIX"              # its synth output, so two deploys never share a lock
export REGION AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"

if [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
  PROFILE=""
  unset AWS_PROFILE
elif [ -n "${PROFILE:-}" ]; then
  export AWS_PROFILE="$PROFILE"
fi
