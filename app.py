#!/usr/bin/env python3
"""CDK application entry point.

A general-purpose agent on Bedrock AgentCore Runtime, reachable from Lark bot chat, with
Lark as the identity provider. Per-user Lark tokens live in the AgentCore Identity Token
Vault (3LO); the agent fetches the calling user's token and passes it to the tool servers
in a custom header, holding no downstream credential of its own.

Deployment is hybrid:
  CDK:  security, agentcore base (role/ECR/S3), shim, router, gateway, observability,
        and sandbox when files_storage=true
  CLI:  the AgentCore Runtimes, Memory, the OAuth provider and the Web Search gateway,
        created by deploy.sh through control-plane APIs; ids fed back via .cdk-state.<name>.json
"""

import json
import os
import pathlib

import aws_cdk as cdk

from stacks.security_stack import SecurityStack
from stacks.agentcore_stack import AgentCoreStack
from stacks.router_stack import RouterStack
from stacks.gateway_stack import GatewayStack
from stacks.shim_stack import ShimStack
from stacks.observability_stack import ObservabilityStack
from stacks.sandbox_stack import SandboxStack
from stacks.webui_stack import WebUiStack

app = cdk.App()

ctx = app.node.try_get_context

# DEPLOYMENT_NAME from .env, passed as -c deployment_name; every resource name starts with it.
prefix = ctx("deployment_name")
if not prefix:
    raise SystemExit("deployment_name context is missing — set DEPLOYMENT_NAME in .env and deploy with ./deploy.sh")

# Per-deployment state (runtime/gateway ids) lives outside version control, one file per
# deployment name — deploy.sh writes it, we inject it so stacks read it via try_get_context.
_state = pathlib.Path(__file__).parent / f".cdk-state.{prefix}.json"
if _state.is_file():
    for k, v in json.loads(_state.read_text()).items():
        if v:
            app.node.set_context(k, v)

env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION") or "us-west-2",
)
# From .env via the deploy scripts (-c lark_api_domain=...). preflight.sh requires it when
# Lark is configured; a deployment without Lark never calls it, and gets Lark's default.
lark_api_domain = ctx("lark_api_domain") or "https://open.larksuite.com"


# Context values arrive as strings when passed with -c, and the string "false" is truthy,
# so compare rather than test.
def _flag(name: str) -> bool:
    return str(ctx(name)).strip().lower() == "true"

# --- Security: Cognito user pool + Secrets Manager slots ---
# The web page's origin travels through .cdk-state.<name>.json: CloudFront assigns it, and
# the hosted login's callback, the router's CORS and the shim all need it.
web_origin = ctx("web_origin") or ""
security = SecurityStack(app, f"{prefix}-security", webui=_flag("webui"),
                         web_origin=web_origin, env=env)

# --- AgentCore base: execution role, ECR image, S3 user files ---
# Runtime itself is created out-of-band by deploy.sh; runtime_arn is derived
# from cdk.json context (runtime_id) once it exists.
agentcore = AgentCoreStack(
    app,
    f"{prefix}-agentcore",
    cognito_user_pool_id=security.user_pool_id,
    cognito_client_id=security.user_pool_client_id,
    cognito_issuer_url=security.cognito_issuer_url,
    cognito_password_secret_name=security.cognito_password_secret.secret_name,
    lark_secret_name=security.lark_secret.secret_name,
    env=env,
)

# --- Shim: Lark OAuth RFC-6749 façade + 3LO return-url ---
# Created before the router so the router can consume its return_url for the
# consent-wait vault check.
shim = ShimStack(
    app,
    f"{prefix}-shim",
    lark_api_domain=lark_api_domain,
    lark_secret_name=security.lark_secret.secret_name,
    web_url=web_origin,
    env=env,
)

# --- Sandbox: the code sandbox's VPC, role and per-user S3 Files workspace (opt-in) ---
# Off unless files_storage=true. Only the Code Interpreter session sits in this VPC and it
# reaches S3 through a free gateway endpoint; nat_gateway=true adds a NAT for internet —
# see docs/architecture.md. Declared before the router so its ids are available to the
# phases that need them.
sandbox = None
if _flag("files_storage"):
    sandbox = SandboxStack(
        app,
        f"{prefix}-sandbox",
        user_files_bucket=agentcore.user_files_bucket,
        nat_gateway=_flag("nat_gateway"),
        env=env,
    )

# --- Router: Lark webhook ingestion (HTTP API + Lambda + DynamoDB identity) ---
router = RouterStack(
    app,
    f"{prefix}-router",
    # The page's origin, so it may trade an h5 code for a JWT. Travels through
    # .cdk-state.<name>.json because CloudFront assigns the domain — the same route the file
    # system id takes, and it breaks the cycle between this stack and the WebUI one.
    web_allowed_origins=web_origin,
    # Only when the page is deployed: the web routes sit behind a JWT authorizer for it.
    web_client_id=security.web_client.user_pool_client_id if security.web_client else "",
    cognito_issuer_url=security.cognito_issuer_url,
    lark_enabled=_flag("lark"),
    runtime_arn=agentcore.runtime_arn,
    runtime_endpoint_qualifier=ctx("runtime_endpoint_id") or "DEFAULT",
    lark_api_domain=lark_api_domain,
    lark_secret_name=security.lark_secret.secret_name,
    shim_return_url=shim.return_url,
    user_pool_id=security.user_pool_id,
    user_pool_arn=security.user_pool_arn,
    user_pool_client_id=security.user_pool_client_id,
    cognito_password_secret_name=security.cognito_password_secret.secret_name,
    env=env,
)

# --- WebUI: static web chat on S3 + CloudFront (opt-in) ---
# Off unless webui=true. Outside Lark a user signs in at the pool's hosted login; inside the
# Lark client the page uses Lark's h5 sign-in, which needs an h5 app registered on the domain.
webui = None
if _flag("webui"):
    webui = WebUiStack(
        app,
        f"{prefix}-webui",
        # From .cdk-state.<name>.json, not router.api_url: a CDK reference makes the router
        # unable to change an export this stack imports (UPDATE_ROLLBACK, measured), the
        # same deadlock the sandbox stack hit.
        router_api_url=ctx("router_api_url") or "",
        # From .env, which deploy.sh sources — the page needs it for requestAuthCode. Empty
        # when the deployment has no Lark: the page then never loads Lark's SDK.
        lark_app_id=os.environ.get("LARK_APP_ID", ""),
        # Through state, like router_api_url, so this stack holds no reference to security.
        web_client_id=ctx("web_client_id") or "",
        web_login_domain=ctx("web_login_domain") or "",
        env=env,
    )

# --- Gateway: demo tool Lambda + Gateway IAM (mcpServer target wired in Phase 3) ---
gateway = GatewayStack(app, f"{prefix}-gateway", env=env)

# --- Observability: dashboard + alarms ---
observability = ObservabilityStack(app, f"{prefix}-observability", env=env)

app.synth()
