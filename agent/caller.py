"""Who is calling: the actor a verified Cognito access token names.

Two app clients of one pool issue tokens, and the client decides the actor's namespace —
Cognito signs `client_id`, so a caller cannot choose it. The router's client (admin auth,
reachable only with IAM) is used for Lark users, named `lark:{open_id}` as their username;
the web client (the hosted login) for everyone else, named `web:{sub}` because an email can
change and `sub` cannot.

The Runtime's CUSTOM_JWT authorizer has checked a token before the request arrives, but
which one it checked when a request carries more than one Authorization header is not
something we have measured, so the container verifies the token it reads itself rather
than decode what was forwarded. The actor the payload names is then only compared with it.
"""

from __future__ import annotations

import logging
import os

import jwt

log = logging.getLogger("agent.caller")

_REGION = os.environ.get("AWS_REGION", "us-west-2")
_POOL = os.environ.get("COGNITO_USER_POOL_ID", "")
_CLIENT = os.environ.get("COGNITO_CLIENT_ID", "")              # the router's
_WEB_CLIENT = os.environ.get("COGNITO_WEB_CLIENT_ID", "")      # the hosted login's; unset = no web login
_ISSUER = f"https://cognito-idp.{_REGION}.amazonaws.com/{_POOL}" if _POOL else ""
_jwks: jwt.PyJWKClient | None = None   # created on first use: no network at import (V2)


def _keys() -> jwt.PyJWKClient:
    global _jwks
    if _jwks is None:
        _jwks = jwt.PyJWKClient(f"{_ISSUER}/.well-known/jwks.json", cache_keys=True,
                                timeout=5)
    return _jwks


def actor(authorization: list[str]) -> str:
    """The actor named by the request's Authorization headers, or "" when there is not
    exactly one, or its token does not verify as an access token this pool issued to one of
    this deployment's clients."""
    return identify(authorization)[0]


def identify(authorization: list[str]) -> tuple[str, str]:
    """(actor, account) for a verified bearer, else ("", ""). The account is what the person
    signed in as — a web user's username, which is their email — for the agent to tell them;
    a Lark user's is read from Lark, with their grant."""
    scheme, _, token = (authorization[0] if len(authorization) == 1 else "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip() or not _ISSUER:
        return "", ""
    token = token.strip()
    try:
        key = _keys().get_signing_key_from_jwt(token).key
        # Leeway for a clock behind Cognito's: a token minted and sent at once would read
        # as issued in the future, and on V2 the container's clock is not established.
        claims = jwt.decode(token, key, algorithms=["RS256"], issuer=_ISSUER, leeway=60,
                            options={"require": ["exp", "iss", "token_use", "client_id",
                                                 "sub"]})
    except Exception as e:  # noqa: BLE001 — any failure is an unverified caller
        log.warning("bearer did not verify: %s", type(e).__name__)
        return "", ""
    if claims["token_use"] != "access":
        log.warning("bearer is not an access token")
        return "", ""
    username = claims.get("username") or ""
    if claims["client_id"] == _CLIENT:
        # Only the router mints for this client, and only for Lark users; anything else
        # holding one of its tokens is no actor at all.
        return (username, "") if username.startswith("lark:") else ("", "")
    if _WEB_CLIENT and claims["client_id"] == _WEB_CLIENT:
        # A Lark user's Cognito account signing in at the hosted login would carry their sub,
        # and with it whatever grants are keyed by it. Lark users come in through Lark.
        if username.startswith("lark:"):
            log.warning("a Lark user's account signed in through the web client")
            return "", ""
        return f"web:{claims['sub']}", username
    log.warning("bearer is not from this deployment's clients")
    return "", ""
