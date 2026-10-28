# shellcheck shell=bash
# The agent Runtime's network (AGENT_NETWORK=public|vpc), for the scripts that deploy or check it.
# Source after lib/env.sh. See docs/architecture.md.

AGENT_NETWORK="${AGENT_NETWORK:-public}"
case "$AGENT_NETWORK" in public|vpc) ;; *)
  echo "AGENT_NETWORK must be public or vpc, not '$AGENT_NETWORK'" >&2; exit 1 ;;
esac
# Comma-separated, spaces tolerated.
AGENT_SUBNET_IDS="${AGENT_SUBNET_IDS:-}"; AGENT_SUBNET_IDS="${AGENT_SUBNET_IDS// /}"
AGENT_SECURITY_GROUP_IDS="${AGENT_SECURITY_GROUP_IDS:-}"; AGENT_SECURITY_GROUP_IDS="${AGENT_SECURITY_GROUP_IDS// /}"
if [ -n "$AGENT_SUBNET_IDS$AGENT_SECURITY_GROUP_IDS" ]; then
  [ "$AGENT_NETWORK" = "vpc" ] || { echo "AGENT_SUBNET_IDS / AGENT_SECURITY_GROUP_IDS only apply with AGENT_NETWORK=vpc" >&2; exit 1; }
  [ -n "$AGENT_SUBNET_IDS" ] && [ -n "$AGENT_SECURITY_GROUP_IDS" ] || {
    echo "AGENT_SUBNET_IDS and AGENT_SECURITY_GROUP_IDS go together: your subnets, your security groups" >&2; exit 1; }
fi
# What the deployment's network is: public, vpc (a VPC this deploy creates) or vpc:byo (yours).
# The three cannot turn into one another in place, so this — not AGENT_NETWORK alone — is locked.
NETWORK_KIND="$AGENT_NETWORK"
[ -z "$AGENT_SUBNET_IDS" ] || NETWORK_KIND="vpc:byo"

# The AZ IDs AgentCore accepts a VPC subnet in, per region (devguide "agentcore-vpc", read
# 2026-10-29). A subnet anywhere else fails at Runtime creation.
supported_az_ids() {
  case "$1" in
    us-east-1) echo "use1-az1 use1-az2 use1-az4" ;;     us-east-2) echo "use2-az1 use2-az2 use2-az3" ;;
    us-west-1) echo "usw1-az1 usw1-az3" ;;               us-west-2) echo "usw2-az1 usw2-az2 usw2-az3" ;;
    ca-central-1) echo "cac1-az1 cac1-az2 cac1-az4" ;;   sa-east-1) echo "sae1-az1 sae1-az2 sae1-az3" ;;
    eu-central-1) echo "euc1-az1 euc1-az2 euc1-az3" ;;   eu-north-1) echo "eun1-az1 eun1-az2 eun1-az3" ;;
    eu-south-1) echo "eus1-az1 eus1-az2 eus1-az3" ;;     eu-south-2) echo "eus2-az1 eus2-az2 eus2-az3" ;;
    eu-west-1) echo "euw1-az1 euw1-az2 euw1-az3" ;;      eu-west-2) echo "euw2-az1 euw2-az2 euw2-az3" ;;
    eu-west-3) echo "euw3-az1 euw3-az2 euw3-az3" ;;      ap-south-1) echo "aps1-az1 aps1-az2 aps1-az3" ;;
    ap-south-2) echo "aps2-az1 aps2-az2 aps2-az3" ;;     ap-northeast-1) echo "apne1-az1 apne1-az2 apne1-az4" ;;
    ap-northeast-2) echo "apne2-az1 apne2-az2 apne2-az3" ;;
    ap-southeast-1) echo "apse1-az1 apse1-az2 apse1-az3" ;; ap-southeast-2) echo "apse2-az1 apse2-az2 apse2-az3" ;;
    ap-southeast-5) echo "apse5-az1 apse5-az2 apse5-az3" ;; ap-southeast-7) echo "apse7-az1 apse7-az2 apse7-az3" ;;
  esac
}

# Two AZ names in this account whose IDs AgentCore supports, comma-separated. Names map to
# different IDs per account, which is why the IDs are what is matched.
supported_az_names() {
  local ids
  ids="$(supported_az_ids "$REGION")"
  [ -n "$ids" ] || return 0
  aws ec2 describe-availability-zones --filters Name=zone-type,Values=availability-zone \
      --query 'AvailabilityZones[].[ZoneId,ZoneName]' --output text \
    | awk -v ok=" $ids " 'index(ok, " " $1 " ") { print $2 }' | sort | head -2 | paste -sd, -
}

# The network a deployment was created with is kept for its life: Code Interpreter and the
# S3 Files mount targets cannot move VPC in place. A state file without the key predates the
# option, so it is public.
network_check_lock() {
  [ -f "$STATE_FILE" ] || return 0
  local recorded
  recorded="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("agent_network") or "public")' "$STATE_FILE")"
  [ "$recorded" = "$NETWORK_KIND" ] || {
    echo "$DEPLOYMENT_NAME was deployed with network '$recorded'; .env now asks for '$NETWORK_KIND' (vpc:byo = AGENT_SUBNET_IDS set)." >&2
    echo "The network is fixed for a deployment's life: set it back, or destroy and redeploy (scripts/destroy.sh)." >&2
    exit 1; }
}
