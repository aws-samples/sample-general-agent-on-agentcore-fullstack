"""Which Lark documents and folders the agent may touch: a Cedar policy store in Amazon Verified
Permissions, checked by the lark-cli MCP server before every document call.

Rules are template-linked policies, created by the router when a user or an admin protects a
resource; no Cedar text is ever built from input. The baseline permits everything, because
Lark's own permissions already bound what a user's token can reach; the rules only take away.
See docs/architecture.md.
"""
from __future__ import annotations

import json

from aws_cdk import (
    CfnOutput,
    Stack,
    aws_iam as iam,
    aws_verifiedpermissions as avp,
)
from constructs import Construct

_SCHEMA = {"Lark": {
    "entityTypes": {
        # Scoped to the app: "<app_id>/<open_id>", because an open_id is per app.
        "User": {"shape": {"type": "Record", "attributes": {}}},
        "Folder": {"memberOfTypes": ["Folder"], "shape": {"type": "Record", "attributes": {}}},
        "Document": {"memberOfTypes": ["Folder"], "shape": {"type": "Record", "attributes": {}}},
    },
    "actions": {
        "read": {"appliesTo": {"principalTypes": ["User"], "resourceTypes": ["Document", "Folder"]}},
        "list": {"appliesTo": {"principalTypes": ["User"], "resourceTypes": ["Folder"]}},
        # Any lark_api call into a document family other than a GET: edits, creates in a
        # folder, moves, copies, permission changes.
        "write": {"appliesTo": {"principalTypes": ["User"], "resourceTypes": ["Document", "Folder"]}},
    },
}}


class ProtectionStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *,
                 enforcer_role: iam.IRole, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.store = avp.CfnPolicyStore(
            self, "Store",
            description="Lark resources the agent may not touch",
            validation_settings=avp.CfnPolicyStore.ValidationSettingsProperty(mode="STRICT"),
            schema=avp.CfnPolicyStore.SchemaDefinitionProperty(cedar_json=json.dumps(_SCHEMA)),
            # Deleting the store deletes every rule with it.
            deletion_protection=avp.CfnPolicyStore.DeletionProtectionProperty(mode="ENABLED"),
        )
        store_id = self.store.attr_policy_store_id

        avp.CfnPolicy(
            self, "Baseline", policy_store_id=store_id,
            definition=avp.CfnPolicy.PolicyDefinitionProperty(
                static=avp.CfnPolicy.StaticPolicyDefinitionProperty(
                    description="Lark's own permissions already bound what a user's token reaches",
                    statement="permit(principal, action, resource);")))
        self.admin_template = avp.CfnPolicyTemplate(
            self, "AdminRule", policy_store_id=store_id,
            description="An admin's rule: nobody's agent may touch this resource or anything beneath it",
            statement="forbid(principal, action, resource in ?resource);")
        self.user_template = avp.CfnPolicyTemplate(
            self, "UserRule", policy_store_id=store_id,
            description="A user's rule: this account's agent may not touch this resource or anything beneath it",
            statement="forbid(principal == ?principal, action, resource in ?resource);")

        # The checker reads, never writes: no tool, and so no model, can change a rule. The role
        # is the execution role every MCP Runtime shares; the writes belong to the router alone.
        iam.Policy(self, "EnforcerRead", roles=[enforcer_role], statements=[iam.PolicyStatement(
            actions=["verifiedpermissions:IsAuthorized", "verifiedpermissions:BatchIsAuthorized",
                     "verifiedpermissions:ListPolicies"],
            resources=[self.store.attr_arn])])

        CfnOutput(self, "PolicyStoreId", value=store_id)
        CfnOutput(self, "PolicyStoreArn", value=self.store.attr_arn)
        CfnOutput(self, "AdminTemplateId", value=self.admin_template.attr_policy_template_id)
        CfnOutput(self, "UserTemplateId", value=self.user_template.attr_policy_template_id)
