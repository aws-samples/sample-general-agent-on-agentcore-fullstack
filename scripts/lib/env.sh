# The deployment target, set up the same way for every script. Source it first; it cds to
# the repo root.
#
#   - .env is the config; a variable already set in the environment wins over it, so
#     `REGION=us-east-1 ./deploy.sh` works and survives into every step deploy.sh runs.
#   - REGION is required and reaches every client: the AWS CLI reads AWS_REGION, boto3 reads
#     only AWS_DEFAULT_REGION (and otherwise falls back to the profile's region — measured,
#     it sent the S3 Files cleanup to another region).
#   - Credentials already in the environment outrank .env's PROFILE.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PREFIX="agentcore-fullstack"

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
export REGION AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"

if [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
  PROFILE=""
  unset AWS_PROFILE
elif [ -n "${PROFILE:-}" ]; then
  export AWS_PROFILE="$PROFILE"
fi
