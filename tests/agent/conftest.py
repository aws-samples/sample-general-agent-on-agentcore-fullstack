"""Puts agent/ on the path and supplies the env a deploy injects: the modules read it at import
and have no fallback."""

import os
import sys
from unittest import mock

import pytest

AGENT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "agent"))
sys.path.insert(0, AGENT)
os.environ.setdefault("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-5")
os.environ.setdefault("LARK_API_DOMAIN", "https://open.larksuite.com")
os.environ.setdefault("LARK_OAUTH_PROVIDER", "test-3lo")
os.environ.setdefault("AGENT_WORKLOAD_NAME", "test-wl")
os.environ.setdefault("METRICS_NAMESPACE", "test/agent")

_POOL_ISSUER = "https://cognito-idp.ap-northeast-1.amazonaws.com/pool-1"


@pytest.fixture
def _signer():
    """A key pair standing in for the pool's, and a way to mint tokens with it."""
    import jwt as pyjwt
    import time as _t
    from identity import caller
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    keys = mock.Mock()
    keys.get_signing_key_from_jwt.return_value = mock.Mock(key=key.public_key())

    def mint(signing_key=key, **over):
        claims = {"iss": _POOL_ISSUER, "token_use": "access", "client_id": "client-1",
                  "username": "lark:ou_alice", "sub": "sub-alice",
                  "exp": int(_t.time()) + 600, **over}
        return "Bearer " + pyjwt.encode(claims, signing_key, algorithm="RS256")
    with mock.patch.object(caller, "_ISSUER", _POOL_ISSUER), \
         mock.patch.object(caller, "_CLIENT", "client-1"), \
         mock.patch.object(caller, "_WEB_CLIENT", "web-client"), \
         mock.patch.object(caller, "_keys", return_value=keys):
        yield mint
