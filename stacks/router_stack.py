"""Router stack — Lark webhook ingestion (HTTP API + Lambda + DynamoDB identity).

Owns the DynamoDB identity table (channel→user mapping, allowlist, sessions, a Lark user's
shared token). Exposes explicit routes:
  POST /webhook/lark   — Lark event subscription callback (only with Lark)
  GET  /health         — health probe
  POST /web/h5/*       — the page inside Lark; a Lark h5 code is the proof
  POST /web/*          — the page in a browser; behind a JWT authorizer for the web client
Signature verification + AES event decryption happen inside the Lambda
(fail-closed). Processing is async: the sync path validates + returns 200 fast,
then self-invokes for the actual agent call.
"""

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_apigatewayv2 as apigwv2,
    aws_apigatewayv2_authorizers as apigwv2_authorizers,
    aws_apigatewayv2_integrations as apigwv2_integrations,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_logs as logs,
)
from constructs import Construct

from stacks import retention_days, lambda_asset


class RouterStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        runtime_arn: str,
        runtime_endpoint_qualifier: str,
        lark_secret_name: str,
        lark_api_domain: str,
        shim_return_url: str,
        user_pool_id: str,
        user_pool_arn: str,
        user_pool_client_id: str,
        cognito_password_secret_name: str,
        web_allowed_origins: str = "",
        web_client_id: str = "",
        cognito_issuer_url: str = "",
        lark_enabled: bool = True,
        protection=None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        region = Stack.of(self).region
        account = Stack.of(self).account
        prefix = self.node.get_context("deployment_name")
        log_retention = self.node.try_get_context("cloudwatch_log_retention_days") or 30
        timeout = int(self.node.try_get_context("lambda_timeout_seconds") or "60")
        memory = int(self.node.try_get_context("lambda_memory_mb") or "256")
        registration_open = str(self.node.try_get_context("registration_open") or "false").lower()
        fn_name = f"{prefix}-router"

        # --- DynamoDB identity table (shared with WebUI stack) ---
        self.identity_table = dynamodb.Table(
            self,
            "IdentityTable",
            table_name=f"{prefix}-identity",
            partition_key=dynamodb.Attribute(name="PK", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="SK", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.DESTROY,  # PoC
        )
        self.identity_table_name = self.identity_table.table_name
        self.identity_table_arn = self.identity_table.table_arn

        log_group = logs.LogGroup(
            self,
            "RouterLogGroup",
            log_group_name=f"/{prefix}/lambda/router",
            retention=retention_days(log_retention),
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.router_fn = _lambda.Function(
            self,
            "RouterFn",
            function_name=fn_name,
            runtime=_lambda.Runtime.PYTHON_3_13,
            architecture=_lambda.Architecture.ARM_64,  # matches aarch64 wheels bundled by uv
            handler="index.handler",
            code=lambda_asset("lambda/router"),
            timeout=Duration.seconds(timeout),
            memory_size=memory,
            environment={
                "AGENTCORE_RUNTIME_ARN": runtime_arn,
                "AGENTCORE_QUALIFIER": runtime_endpoint_qualifier,
                "IDENTITY_TABLE_NAME": self.identity_table.table_name,
                "LARK_SECRET_ID": lark_secret_name,
                "LARK_API_DOMAIN": lark_api_domain,
                "REGISTRATION_OPEN": registration_open,
                "SELF_FUNCTION_NAME": fn_name,
                # Without this the code assumed 60s and cut the agent off early,
                # leaving part of the Lambda's budget unused.
                "LAMBDA_TIMEOUT_SECONDS": str(timeout),
                # Empty without Lark: then there is no Lark provider to offer.
                "LARK_OAUTH_PROVIDER": f"{prefix}-3lo" if lark_enabled else "",
                "AGENT_WORKLOAD_NAME": f"{prefix}-wl",
                "LARK_SCOPES": "drive:drive docx:document offline_access",
                # Per-IdP registry for /auth; written by scripts/setup-3lo.sh.
                "IDP_REGISTRY": self.node.try_get_context("idp_registry") or "",
                "SHIM_RETURN_URL": shim_return_url,
                # Token factory: the router signs each turn's identity (see cognito.py).
                "COGNITO_USER_POOL_ID": user_pool_id,
                "COGNITO_CLIENT_ID": user_pool_client_id,
                "COGNITO_PASSWORD_SECRET_ID": cognito_password_secret_name,
                # The hosted login's client: its tokens are web users' (see index.py).
                "COGNITO_WEB_CLIENT_ID": web_client_id,
                # Where protection rules are written (stacks/protection_stack.py); empty without Lark.
                "POLICY_STORE_ID": protection.store.attr_policy_store_id if protection else "",
                "PROTECT_ADMIN_TEMPLATE_ID": protection.admin_template.attr_policy_template_id if protection else "",
                "PROTECT_USER_TEMPLATE_ID": protection.user_template.attr_policy_template_id if protection else "",
                # The Web Search gateway's blocked domains (websearch.py), by the names provision.sh gives.
                "WEB_SEARCH_GATEWAY_NAME": f"{prefix}-websearch-gw",
                "WEB_SEARCH_POLICY_ENGINE_NAME": f"{prefix.replace('-', '_')}_websearch",
                "WEB_SEARCH_REGION": "us-east-1",
            },
            log_group=log_group,
        )

        # The sync phase self-invokes asynchronously, and Lambda counts a timeout as
        # a function error — with the default 2 retries a slow turn would be replayed
        # twice, so the agent would redo the work and the user would get duplicate
        # replies. Deliver once; the user can always ask again.
        self.router_fn.configure_async_invoke(retry_attempts=0)

        integration = apigwv2_integrations.HttpLambdaIntegration("Integration", handler=self.router_fn)
        # The web page's /web/session call is a cross-origin JSON POST, so the browser sends
        # a preflight first. Without this it 404s and the page reports only "Failed to fetch".
        cors = apigwv2.CorsPreflightOptions(
            allow_origins=[o.strip() for o in web_allowed_origins.split(",") if o.strip()],
            allow_methods=[apigwv2.CorsHttpMethod.GET, apigwv2.CorsHttpMethod.POST,
                           apigwv2.CorsHttpMethod.DELETE, apigwv2.CorsHttpMethod.OPTIONS],
            allow_headers=["content-type", "authorization"],
        ) if web_allowed_origins else None
        self.http_api = apigwv2.HttpApi(
            self, "RouterApi", api_name=f"{prefix}-router",
            description="Lark webhook ingestion (explicit routes only)",
            cors_preflight=cors,
        )
        if lark_enabled:
            self.http_api.add_routes(
                path="/webhook/lark", methods=[apigwv2.HttpMethod.POST], integration=integration,
            )
        self.http_api.add_routes(
            path="/health", methods=[apigwv2.HttpMethod.GET], integration=integration,
        )
        # The page inside Lark: an h5 authorization code in, the user's token out; and the
        # chat commands, here rather than on the Runtime because they rotate ids the router owns.
        if lark_enabled:
            for path in ("/web/h5/session", "/web/h5/command"):
                self.http_api.add_routes(path=path, methods=[apigwv2.HttpMethod.POST],
                                         integration=integration)
        # The page in a browser: the hosted login's access token, verified here before the
        # handler reads its sub, and forwarded to the Runtime, which verifies it again.
        if web_client_id:
            authorizer = apigwv2_authorizers.HttpJwtAuthorizer(
                "WebLogin", cognito_issuer_url, jwt_audience=[web_client_id],
                identity_source=["$request.header.Authorization"])
            for path in ("/web/session", "/web/command", "/web/consent"):
                self.http_api.add_routes(path=path, methods=[apigwv2.HttpMethod.POST],
                                         integration=integration, authorizer=authorizer)
            # Protection rules: the caller's own, and (admins group, checked by the handler)
            # the ones that hold for everyone.
            for path in ("/web/protection", "/web/admin/protection", "/web/admin/websearch"):
                self.http_api.add_routes(
                    path=path,
                    methods=[apigwv2.HttpMethod.GET, apigwv2.HttpMethod.POST, apigwv2.HttpMethod.DELETE],
                    integration=integration, authorizer=authorizer)
            # The admin page; the handler checks the admins group on the verified token.
            self.http_api.add_routes(
                path="/web/admin/users",
                methods=[apigwv2.HttpMethod.GET, apigwv2.HttpMethod.POST, apigwv2.HttpMethod.DELETE],
                integration=integration, authorizer=authorizer)

        # throttling on default stage
        default_stage = self.http_api.default_stage
        if default_stage:
            cfn_stage = default_stage.node.default_child
            cfn_stage.default_route_settings = apigwv2.CfnStage.RouteSettingsProperty(
                throttling_burst_limit=20,
                throttling_rate_limit=50,
                detailed_metrics_enabled=True,
            )

        # --- IAM ---
        # No InvokeAgentRuntime here: the runtime is reached over HTTPS with the user's
        # own JWT, and a CUSTOM_JWT runtime refuses SigV4 anyway. Authorization for that
        # hop is the token, not this role.
        self.identity_table.grant_read_write_data(self.router_fn)
        # The only writer of protection rules: the commands and the page come through here,
        # and no tool does. Rules are template-linked, so this role never writes Cedar text.
        if protection:
            self.router_fn.add_to_role_policy(iam.PolicyStatement(
                actions=["verifiedpermissions:CreatePolicy", "verifiedpermissions:DeletePolicy",
                         "verifiedpermissions:GetPolicy", "verifiedpermissions:ListPolicies"],
                resources=[protection.store.attr_arn]))

        # The Web Search gateway's blocked domains: its target's exclude list and the forbid in
        # its policy engine. Never the gateway itself, so no PassRole and no mode change.
        ws = f"arn:aws:bedrock-agentcore:us-east-1:{account}"
        rt_prefix = prefix.replace("-", "_")
        self.router_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock-agentcore:ListGateways", "bedrock-agentcore:ListPolicyEngines"], resources=["*"]))
        self.router_fn.add_to_role_policy(iam.PolicyStatement(
            # Create/UpdatePolicy also need ManageResourceScopedPolicy and InvokeGateway on the gateway a
            # policy names: the engine validates the statement against it as the caller.
            actions=["bedrock-agentcore:GetGateway", "bedrock-agentcore:ListGatewayTargets",
                     "bedrock-agentcore:GetGatewayTarget", "bedrock-agentcore:UpdateGatewayTarget",
                     "bedrock-agentcore:ManageResourceScopedPolicy", "bedrock-agentcore:InvokeGateway"],
            resources=[f"{ws}:gateway/{prefix}-websearch-gw-*"]))
        self.router_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock-agentcore:ListPolicies", "bedrock-agentcore:GetPolicy", "bedrock-agentcore:CreatePolicy",
                     "bedrock-agentcore:UpdatePolicy", "bedrock-agentcore:DeletePolicy"],
            resources=[f"{ws}:policy-engine/{rt_prefix}_websearch-*"]))

        self.router_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["lambda:InvokeFunction"],
                resources=[f"arn:aws:lambda:{region}:{account}:function:{fn_name}"],
            )
        )
        self.router_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"],
                resources=[
                    f"arn:aws:secretsmanager:{region}:{account}:secret:{prefix}/*",
                    # Identity-managed OAuth provider secret (for the vault check below).
                    f"arn:aws:secretsmanager:{region}:{account}:secret:bedrock-agentcore-identity!default/oauth2/*",
                ],
            )
        )
        # 3LO consent-wait: check the vault for the user's Lark token (same
        # USER_FEDERATION sequence the agent uses) so the router can hold and
        # re-invoke once consent completes, sparing the user a re-send. Also completes
        # the consent itself — see complete_consent in the router.
        #
        # ForJWT, and deliberately NOT ForUserId: the router mints the user's JWT, so it
        # needs no by-name path, and the vault namespace follows the token's `sub`.
        # This is the trust the design relocates here — the router is small, runs no
        # model and never sees untrusted input, which the agent cannot claim.
        self.router_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                    "bedrock-agentcore:GetWorkloadAccessToken",
                    "bedrock-agentcore:GetResourceOauth2Token",
                    "bedrock-agentcore:CompleteResourceTokenAuth",
                ],
                resources=["*"],
            )
        )

        # Token factory: mint a per-user Cognito access token, so every downstream hop
        # verifies a signature instead of trusting a string.
        self.router_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "cognito-idp:AdminGetUser",
                    "cognito-idp:AdminCreateUser",
                    "cognito-idp:AdminSetUserPassword",
                    "cognito-idp:AdminInitiateAuth",
                    # The admin page, managing web users.
                    "cognito-idp:ListUsers",
                    "cognito-idp:ListUsersInGroup",
                    "cognito-idp:AdminDeleteUser",
                    "cognito-idp:AdminUserGlobalSignOut",
                ],
                resources=[user_pool_arn],
            )
        )

        # Nothing here for /status or /clear any more: the conversation is LangGraph
        # state, so the agent answers both over InvokeAgentRuntime and the router needs
        # no Memory permissions of its own (dropped ListMemories/ListEvents/DeleteEvent).

        self.api_url = self.http_api.url or ""
        CfnOutput(self, "ApiUrl", value=self.api_url)
        CfnOutput(self, "WebhookLarkUrl", value=(self.http_api.url or "") + "webhook/lark")
        CfnOutput(self, "IdentityTableName", value=self.identity_table.table_name)
