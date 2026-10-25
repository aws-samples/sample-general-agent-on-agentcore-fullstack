#!/usr/bin/env bash
# Web users of the page's hosted login — who may sign in without Lark.
#
#   scripts/manage-users.sh add <email>      create the user; Cognito emails a temporary password
#   scripts/manage-users.sh remove <email>   sign them out everywhere, then delete them
#   scripts/manage-users.sh list             the web users (Lark users are created by the router)
#
# Self sign-up is off, so this is the web's allowlist. Target comes from .env.
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/lib/env.sh"

pool="$(aws cloudformation describe-stacks --stack-name "$PREFIX-security" \
  --query "Stacks[0].Outputs[?OutputKey=='UserPoolId'].OutputValue" --output text)"
[ -n "$pool" ] && [ "$pool" != "None" ] || { echo "no user pool — run ./deploy.sh first"; exit 1; }

case "${1:-}" in
  add)
    email="${2:?usage: $0 add <email>}"
    # The email is the username, so it is what they type at the hosted login.
    aws cognito-idp admin-create-user --user-pool-id "$pool" --username "$email" \
      --user-attributes Name=email,Value="$email" Name=email_verified,Value=true \
      --desired-delivery-mediums EMAIL --query 'User.Username' --output text
    echo "  invited: a temporary password is on its way to $email"
    ;;
  remove)
    email="${2:?usage: $0 remove <email>}"
    # A removed user's access token stays valid until it expires; signing them out first
    # revokes their refresh token, so the page cannot renew it.
    aws cognito-idp admin-user-global-sign-out --user-pool-id "$pool" --username "$email"
    aws cognito-idp admin-delete-user --user-pool-id "$pool" --username "$email"
    echo "  removed $email"
    ;;
  list)
    aws cognito-idp list-users --user-pool-id "$pool" \
      --query "Users[?!starts_with(Username, 'lark:')].[Username,UserStatus]" --output text
    ;;
  *) sed -n '2,8p' "$0" | cut -c3-; exit 1 ;;
esac
