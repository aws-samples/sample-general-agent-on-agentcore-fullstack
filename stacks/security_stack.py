"""Security stack — Cognito user pool + Secrets Manager slots.

One pool, two kinds of user, told apart by the app client their token was issued to:

- **Lark users**: Lark is the IdP. The router creates them as `lark:{open_id}` with an
  HMAC-derived password and mints their tokens through the `app` client, which only allows
  admin auth (IAM). Lark chat and the page opened inside Lark both resolve to that user.
- **Web users** (when the web page is deployed): created by an admin, username = their
  email, their own password, signed in at the pool's hosted login through the `web` client
  (authorization code + PKCE). Their actor is `web:{sub}` (agent/identity/caller.py).

The pool is only ever updated in place: AgentCore Identity keys a user's grants by the
token's `sub`, so a replaced pool would orphan every Lark and Google grant.
"""

from aws_cdk import (
    Stack,
    RemovalPolicy,
    CfnOutput,
    aws_secretsmanager as secretsmanager,
    aws_cognito as cognito,
)
from constructs import Construct


class SecurityStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, webui: bool = False,
                 web_origin: str = "", **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        prefix = self.node.get_context("deployment_name")

        # --- Cognito user pool ---
        # Self sign-up stays off: a web user exists because an admin created them, which is
        # the web's allowlist. Recovery by email lets them reset their own password.
        self.user_pool = cognito.UserPool(
            self,
            "UserPool",
            user_pool_name=f"{prefix}-users",
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(username=True),
            # People choose web users' passwords; a Lark user's is derived (36 characters).
            password_policy=cognito.PasswordPolicy(
                min_length=8,
                require_lowercase=True,
                require_uppercase=False,
                require_digits=True,
                require_symbols=False,
            ),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            removal_policy=RemovalPolicy.DESTROY,  # PoC — recreate freely
        )

        self.user_pool_client = self.user_pool.add_client(
            "AppClient",
            user_pool_client_name=f"{prefix}-app",
            auth_flows=cognito.AuthFlow(admin_user_password=True),
            generate_secret=False,
            # id_token_validity left at Cognito default (1h)
        )

        # --- Hosted login for the web page ---
        # Classic hosted UI: it needs no per-client branding resource. The callback is the
        # page itself, known only once CloudFront assigns it — until then a placeholder, and
        # scripts/provision.sh redeploys this stack with the real origin.
        self.web_client = None
        self.web_login_domain = ""
        if webui:
            page = (web_origin.rstrip("/") + "/") if web_origin else "https://localhost/"
            # Every page that signs in comes back to itself: the chat and the admin page.
            pages = [page, page + "index.html", page + "admin.html"]
            domain = self.user_pool.add_domain(
                "WebLogin",
                # Unique per region across all accounts, hence the account in it.
                cognito_domain=cognito.CognitoDomainOptions(
                    domain_prefix=f"{prefix}-{Stack.of(self).account}"),
            )
            self.web_client = self.user_pool.add_client(
                "WebClient",
                user_pool_client_name=f"{prefix}-web",
                generate_secret=False,                     # a browser cannot keep one
                auth_flows=cognito.AuthFlow(),
                o_auth=cognito.OAuthSettings(
                    flows=cognito.OAuthFlows(authorization_code_grant=True),
                    scopes=[cognito.OAuthScope.OPENID, cognito.OAuthScope.EMAIL],
                    callback_urls=pages, logout_urls=pages),
                supported_identity_providers=[cognito.UserPoolClientIdentityProvider.COGNITO],
                prevent_user_existence_errors=True,
            )
            self.web_login_domain = domain.base_url()
            # Who may open the admin page and manage web users. The first member is added at
            # deploy from WEB_ADMIN_EMAIL (scripts/provision.sh).
            cognito.CfnUserPoolGroup(self, "Admins", user_pool_id=self.user_pool.user_pool_id,
                                     group_name="admins",
                                     description="May manage web users on the admin page")
            CfnOutput(self, "WebClientId", value=self.web_client.user_pool_client_id)
            CfnOutput(self, "WebLoginDomain", value=self.web_login_domain)

        self.user_pool_id = self.user_pool.user_pool_id
        self.user_pool_arn = self.user_pool.user_pool_arn
        self.user_pool_client_id = self.user_pool_client.user_pool_client_id
        self.cognito_issuer_url = (
            f"https://cognito-idp.{Stack.of(self).region}.amazonaws.com/"
            f"{self.user_pool.user_pool_id}"
        )

        # --- Lark app credentials (placeholder — filled by scripts/setup-lark.sh) --
        # JSON shape: {"appId","appSecret","verificationToken","encryptKey"}
        self.lark_secret = secretsmanager.Secret(
            self,
            "LarkSecret",
            secret_name=f"{prefix}/channels/lark",
            description="Lark self-built app credentials "
            "(appId/appSecret/verificationToken/encryptKey)",
            removal_policy=RemovalPolicy.DESTROY,
        )

        # --- HMAC salt for deriving deterministic Cognito passwords ---
        self.cognito_password_secret = secretsmanager.Secret(
            self,
            "CognitoPasswordSecret",
            secret_name=f"{prefix}/cognito-password-secret",
            description="HMAC salt for deriving Cognito user passwords",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                password_length=64,
                exclude_punctuation=True,
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        CfnOutput(self, "UserPoolId", value=self.user_pool_id)
        CfnOutput(self, "UserPoolClientId", value=self.user_pool_client_id)
        CfnOutput(self, "CognitoIssuerUrl", value=self.cognito_issuer_url)
        CfnOutput(self, "LarkSecretName", value=self.lark_secret.secret_name)
