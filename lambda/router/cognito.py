"""Per-user Cognito JWT minting — the router's "token factory".

The router is the only component that turns a Lark `open_id` into a signed identity.
Everything downstream then verifies a signature instead of trusting a string: the
Runtime's CUSTOM_JWT authorizer validates the token on the way in, and AgentCore
Identity keys the Token Vault off the token's `sub`.

Access tokens, not id tokens: `client_id` (which authorizers match on) is only in the
access token, and it is the access token AgentCore derives the vault identity from.

Passwords are HMAC-derived from a Secrets Manager salt, so they are deterministic and
never stored. Only Lark users are minted for: a web user signs in with their own password,
which a reset here would overwrite.

One token per Lark user, shared by every router container through the identity table. A
consent can only be completed with the very token it was started with (measured — see
docs/agentcore-behavior.md), and a per-container cache handed the completion a different
one whenever the consent came back to another container.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger("router.cognito")

_REGION = os.environ.get("AWS_REGION", "us-west-2")
_USER_POOL_ID = os.environ.get("COGNITO_USER_POOL_ID", "")
_CLIENT_ID = os.environ.get("COGNITO_CLIENT_ID", "")
_PASSWORD_SECRET_ID = os.environ.get("COGNITO_PASSWORD_SECRET_ID", "")
_TABLE_NAME = os.environ.get("IDENTITY_TABLE_NAME", "")
# Renewed when less than this remains, so a turn never starts on a token about to expire;
# the one it replaces is kept as `previous` for consents started with it.
_RENEW_SECONDS = 20 * 60

_cognito = boto3.client("cognito-idp", region_name=_REGION)
_secrets = boto3.client("secretsmanager", region_name=_REGION)
_table = boto3.resource("dynamodb", region_name=_REGION).Table(_TABLE_NAME) if _TABLE_NAME else None

_salt: str | None = None


def configured() -> bool:
    return bool(_USER_POOL_ID and _CLIENT_ID and _PASSWORD_SECRET_ID)


def _get_salt() -> str:
    global _salt
    if _salt is None:
        _salt = _secrets.get_secret_value(SecretId=_PASSWORD_SECRET_ID)["SecretString"]
    return _salt


def _password(username: str) -> str:
    digest = hmac.new(_get_salt().encode(), username.encode(), hashlib.sha256).hexdigest()
    return digest[:32] + "Aa1!"   # suffix satisfies Cognito complexity


def _ensure_user(username: str) -> None:
    try:
        _cognito.admin_get_user(UserPoolId=_USER_POOL_ID, Username=username)
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "UserNotFoundException":
            raise
    _cognito.admin_create_user(
        UserPoolId=_USER_POOL_ID, Username=username, MessageAction="SUPPRESS",
        # A colon is invalid in an email local part, and the username carries one.
        UserAttributes=[{"Name": "email", "Value": f"{username.replace(':', '-')}@lark.local"},
                        {"Name": "email_verified", "Value": "true"}],
    )
    _cognito.admin_set_user_password(
        UserPoolId=_USER_POOL_ID, Username=username,
        Password=_password(username), Permanent=True,
    )
    log.info("provisioned cognito user %s", username)


def _exp(token: str) -> float:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload)).get("exp", 0))
    except Exception:  # noqa: BLE001
        return 0.0


def _mint(actor_id: str) -> str:
    _ensure_user(actor_id)
    params = {"USERNAME": actor_id, "PASSWORD": _password(actor_id)}
    try:
        resp = _cognito.admin_initiate_auth(
            UserPoolId=_USER_POOL_ID, ClientId=_CLIENT_ID,
            AuthFlow="ADMIN_USER_PASSWORD_AUTH", AuthParameters=params)
    except ClientError as e:
        # Salt rotated → the derived password no longer matches. Reset and retry once.
        if e.response["Error"]["Code"] not in ("NotAuthorizedException", "UserNotFoundException"):
            raise
        _ensure_user(actor_id)
        _cognito.admin_set_user_password(
            UserPoolId=_USER_POOL_ID, Username=actor_id,
            Password=_password(actor_id), Permanent=True)
        resp = _cognito.admin_initiate_auth(
            UserPoolId=_USER_POOL_ID, ClientId=_CLIENT_ID,
            AuthFlow="ADMIN_USER_PASSWORD_AUTH", AuthParameters=params)
    return resp["AuthenticationResult"]["AccessToken"]


def _key(actor_id: str) -> dict:
    return {"PK": f"TOKEN#{actor_id}", "SK": "LARK"}


def _stored(actor_id: str) -> dict:
    return _table.get_item(Key=_key(actor_id), ConsistentRead=True).get("Item") or {}


def user_jwt(actor_id: str) -> str:
    """The Lark user's shared access token, renewed when under _RENEW_SECONDS remain.

    Renewal is a conditional write on the token it replaces: two containers renewing at once
    must not each go on with a token of their own, or the one that lost holds a token no
    completion will try. The loser re-reads and uses the winner's."""
    if not configured():
        raise RuntimeError("Cognito env not configured "
                           "(COGNITO_USER_POOL_ID / CLIENT_ID / PASSWORD_SECRET_ID)")
    if not actor_id.startswith("lark:"):
        raise ValueError(f"tokens are minted for Lark users only, not {actor_id!r}")
    item = _stored(actor_id)
    if item.get("current") and float(item.get("currentExp", 0)) - time.time() > _RENEW_SECONDS:
        return item["current"]
    token = _mint(actor_id)
    exp = _exp(token)
    # Twice at most: a failed condition is either another container's renewal, whose token is
    # then used, or TTL deleting the expired record between the read and the write — measured
    # on a user idle past the token's hour — after which nobody has won and this one is written.
    for _ in range(2):
        try:
            _table.put_item(
                Item={**_key(actor_id), "current": token, "currentExp": int(exp),
                      "previous": item.get("current", ""), "previousExp": int(item.get("currentExp", 0)),
                      "ttl": int(exp)},
                ConditionExpression=("attribute_not_exists(PK)" if not item.get("current")
                                     else "#c = :seen"),
                **({"ExpressionAttributeNames": {"#c": "current"},
                    "ExpressionAttributeValues": {":seen": item["current"]}}
                   if item.get("current") else {}))
            return token
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
                raise
            item = _stored(actor_id)
            if item.get("current"):
                return item["current"]
    raise RuntimeError(f"could not store a renewed token for {actor_id}")


def consent_tokens(actor_id: str) -> list[str]:
    """The tokens a consent of this Lark user can have been started with, newest first —
    read only: completing with a token minted now would be the one token that cannot work."""
    item = _stored(actor_id)
    now = time.time()
    return [item[k] for k in ("current", "previous")
            if item.get(k) and float(item.get(k + "Exp", 0)) > now]


# --- web users, for the admin page ------------------------------------------------
# Self sign-up is off, so the pool is the web's allowlist, and these are how an admin edits
# it without an AWS console. Lark users are the router's own and never listed or touched.

ADMINS_GROUP = "admins"


def web_users() -> list[dict]:
    """[{email, status, admin}] for the pool's web users — every user not named `lark:…`."""
    admins = {u["Username"] for u in _cognito.list_users_in_group(
        UserPoolId=_USER_POOL_ID, GroupName=ADMINS_GROUP, Limit=60).get("Users", [])}
    users, token = [], None
    while True:
        page = _cognito.list_users(UserPoolId=_USER_POOL_ID, Limit=60,
                                   **({"PaginationToken": token} if token else {}))
        users += [{"email": u["Username"], "status": u.get("UserStatus", ""),
                   "admin": u["Username"] in admins}
                  for u in page.get("Users", []) if not u["Username"].startswith("lark:")]
        token = page.get("PaginationToken")
        if not token:
            return users


def invite(email: str) -> None:
    """Create a web user; Cognito emails them a temporary password. The email is the
    username, so it is what they type at the hosted login."""
    if email.startswith("lark:"):
        raise ValueError("Lark users are not managed here")
    _cognito.admin_create_user(
        UserPoolId=_USER_POOL_ID, Username=email, DesiredDeliveryMediums=["EMAIL"],
        UserAttributes=[{"Name": "email", "Value": email},
                        {"Name": "email_verified", "Value": "true"}])


def remove(email: str) -> None:
    """Sign a web user out everywhere, then delete them. Their access token stays valid until
    it expires; signing out first revokes the refresh token, so no page can renew it."""
    if email.startswith("lark:"):
        raise ValueError("Lark users are not managed here")
    _cognito.admin_user_global_sign_out(UserPoolId=_USER_POOL_ID, Username=email)
    _cognito.admin_delete_user(UserPoolId=_USER_POOL_ID, Username=email)
