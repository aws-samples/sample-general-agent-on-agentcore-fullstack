"""Who is calling: the actor a verified Cognito access token names.

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
_CLIENT = os.environ.get("COGNITO_CLIENT_ID", "")
_ISSUER = f"https://cognito-idp.{_REGION}.amazonaws.com/{_POOL}" if _POOL else ""
_jwks: jwt.PyJWKClient | None = None   # created on first use: no network at import (V2)


def _keys() -> jwt.PyJWKClient:
    global _jwks
    if _jwks is None:
        _jwks = jwt.PyJWKClient(f"{_ISSUER}/.well-known/jwks.json", cache_keys=True,
                                timeout=5)
    return _jwks


def actor(authorization: list[str]) -> str:
    """The actor ("lark:{open_id}") named by the request's Authorization headers, or ""
    when there is not exactly one, or its token does not verify as an access token this
    pool issued to this deployment's client. The username is the actor: the router creates
    each Cognito user under it (lambda/router/cognito.py)."""
    scheme, _, token = (authorization[0] if len(authorization) == 1 else "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip() or not _ISSUER:
        return ""
    token = token.strip()
    try:
        key = _keys().get_signing_key_from_jwt(token).key
        # Leeway for a clock behind Cognito's: a token minted and sent at once would read
        # as issued in the future, and on V2 the container's clock is not established.
        claims = jwt.decode(token, key, algorithms=["RS256"], issuer=_ISSUER, leeway=60,
                            options={"require": ["exp", "iss", "token_use", "client_id",
                                                 "username"]})
    except Exception as e:  # noqa: BLE001 — any failure is an unverified caller
        log.warning("bearer did not verify: %s", type(e).__name__)
        return ""
    if claims["token_use"] != "access" or claims["client_id"] != _CLIENT:
        log.warning("bearer is not this client's access token")
        return ""
    return claims["username"]
