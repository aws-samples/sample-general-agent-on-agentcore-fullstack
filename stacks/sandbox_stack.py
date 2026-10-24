"""The code sandbox: its VPC, its execution role, and the per-user S3 Files workspace
mounted into it.

Only deployed when the `files_storage` context flag is true. What sits in this VPC is the
Code Interpreter session; it reaches S3 through a free gateway endpoint, and the agent
Runtime stays PUBLIC outside it. By default there is **no NAT**, so the sandbox has no
internet; `nat_gateway=true` adds one NAT gateway so it can install packages and call
external APIs.

What isolation rests on (see docs/architecture.md):
  - one Access Point per user, `rootDirectory` fixed server-side to /users/<actor>
  - the session mounts exactly one Access Point, enforced at the microVM boundary
Neither is agent code, which is the point — the agent runs model output.

The Access Points themselves are NOT created here: they are per-user, created on first use
and named in `StartCodeInterpreterSession`.
"""
from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Stack,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_s3 as s3,
    aws_s3files as s3files,
)
from constructs import Construct


# NFS. The mount target listens here and nowhere else.
_NFS_PORT = 2049
# Subnet group the sandbox and mount targets live in. The name is part of each subnet's
# logical id, so it stays "isolated" even when the NAT makes it private-with-egress.
_SANDBOX_SUBNETS = "isolated"


class SandboxStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *,
                 user_files_bucket: s3.IBucket, nat_gateway: bool = False,
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        region = Stack.of(self).region
        account = Stack.of(self).account
        prefix = self.node.get_context("deployment_name")

        # --- VPC for the Code Interpreter session. S3 goes through a gateway endpoint for
        # free; the agent Runtime stays PUBLIC outside it. Two AZs because mount targets
        # are per-AZ. See ADR 0007.
        #
        # The sandbox group is listed first and keeps its name either way, so toggling the
        # NAT retypes the same subnets (same logical ids, same CIDRs) instead of replacing
        # them: the Code Interpreter has their ids baked in and has no update API. Public
        # subnets only exist to hold the NAT; one NAT, not one per AZ, to keep it to one
        # fixed hourly charge.
        sandbox_subnets = ec2.SubnetConfiguration(
            name=_SANDBOX_SUBNETS, cidr_mask=24,
            subnet_type=(ec2.SubnetType.PRIVATE_WITH_EGRESS if nat_gateway
                         else ec2.SubnetType.PRIVATE_ISOLATED))
        self.vpc = ec2.Vpc(
            self, "Vpc",
            max_azs=2,
            nat_gateways=1 if nat_gateway else 0,
            ip_addresses=ec2.IpAddresses.cidr("10.20.0.0/16"),
            subnet_configuration=[sandbox_subnets] + ([ec2.SubnetConfiguration(
                name="public", cidr_mask=24, subnet_type=ec2.SubnetType.PUBLIC)]
                if nat_gateway else []),
        )
        # Same construct id the `gateway_endpoints` option would use, so the endpoint keeps
        # its logical id; added explicitly to hold a handle for its policy below.
        self.s3_endpoint = self.vpc.add_gateway_endpoint(
            "S3", service=ec2.GatewayVpcEndpointAwsService.S3)

        # The sandbox's own security group, and the mount targets' — separated so the file
        # system accepts NFS from the sandbox only, not from anything else in the VPC.
        self.sandbox_sg = ec2.SecurityGroup(
            self, "SandboxSg", vpc=self.vpc, allow_all_outbound=True,
            description="Code Interpreter session: S3 via the gateway endpoint, and the mount")
        self.mount_sg = ec2.SecurityGroup(
            self, "MountTargetSg", vpc=self.vpc, allow_all_outbound=False,
            description="S3 Files mount targets: NFS from the sandbox only")
        self.mount_sg.add_ingress_rule(
            peer=self.sandbox_sg, connection=ec2.Port.tcp(_NFS_PORT),
            description="NFS from the Code Interpreter session")

        # --- The file system over the user-files bucket, and the role it uses to move
        # data in and out of that bucket.
        # Trusted by elasticfilesystem, NOT s3files: S3 Files runs on the EFS control
        # plane (hence mount.s3files coming from amazon-efs-utils and the
        # elasticfilesystem:Client* aliases), and IAM rejects s3files.amazonaws.com as an
        # unknown principal. The conditions keep the trust to file systems in this
        # account and region, since the principal itself is a whole other service.
        self.fs_role = iam.Role(
            self, "FileSystemRole",
            assumed_by=iam.ServicePrincipal(
                "elasticfilesystem.amazonaws.com",
                conditions={
                    "StringEquals": {"aws:SourceAccount": account},
                    "ArnLike": {
                        "aws:SourceArn":
                            f"arn:aws:s3files:{region}:{account}:file-system/*"},
                }),
            description="Lets S3 Files move data between the file system and the bucket")
        # Whole bucket, not just users/*: the file system spans the bucket, and each
        # user's confinement comes from their Access Point instead. This role is assumable
        # only by the service, never by the agent.
        #
        # Ported from the reference implementation rather than left to grant_read_write:
        # the service also needs ListBucketVersions, and it manages EventBridge rules for
        # bucket/file-system synchronisation under a fixed name prefix. The ResourceAccount
        # conditions keep the role from reaching a bucket in another account.
        user_files_bucket.grant_read_write(self.fs_role)
        self.fs_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:ListBucket", "s3:ListBucketVersions"],
            resources=[user_files_bucket.bucket_arn],
            conditions={"StringEquals": {"aws:ResourceAccount": account}}))
        self.fs_role.add_to_policy(iam.PolicyStatement(
            actions=["events:PutRule", "events:PutTargets", "events:DeleteRule",
                     "events:DisableRule", "events:EnableRule", "events:RemoveTargets"],
            resources=["arn:aws:events:*:*:rule/DO-NOT-DELETE-S3-Files*"],
            conditions={"StringEquals": {
                "events:ManagedBy": "elasticfilesystem.amazonaws.com"}}))
        self.fs_role.add_to_policy(iam.PolicyStatement(
            actions=["events:DescribeRule", "events:ListRules",
                     "events:ListRuleNamesByTarget", "events:ListTargetsByRule"],
            resources=["arn:aws:events:*:*:rule/*"]))

        # `prefix` is deliberately not set. The layout is decided in exactly one place —
        # each Access Point's rootDirectory (/users/<actor>) — and scoping the file
        # system to a prefix as well would stack the two, with semantics we cannot
        # confirm without deploying. One source of truth is worth more here.
        self.file_system = s3files.CfnFileSystem(
            self, "FileSystem",
            # The bucket ARN, not its name: CloudFormation validates this against
            # ^(arn:aws[a-zA-Z0-9-]*:s3:::.+)$ and rejects a bare name.
            bucket=user_files_bucket.bucket_arn,
            role_arn=self.fs_role.role_arn,
            # The bucket is ours and holds only what this agent wrote, so the warning
            # about pointing a file system at an existing bucket is expected.
            accept_bucket_warning=True,
        )
        # A resource-policy Deny that no identity policy can override: mounting is refused
        # whenever s3files:AccessPointArn is absent, i.e. nobody may mount the file system
        # root. The STS session policy already scopes each session's credentials to one
        # Access Point; this closes the root escape structurally, so a later mistake in an
        # identity policy cannot reopen it. Ported from the reference implementation.
        s3files.CfnFileSystemPolicy(
            self, "FileSystemPolicy",
            file_system_id=self.file_system.ref,
            policy={
                "Version": "2012-10-17",
                "Statement": [{
                    "Sid": "DenyMountWithoutAccessPoint",
                    "Effect": "Deny",
                    "Principal": "*",
                    "Action": ["s3files:ClientMount", "s3files:ClientWrite",
                               "s3files:ClientRootAccess"],
                    "Condition": {"Null": {"s3files:AccessPointArn": "true"}},
                }],
            },
        )

        for i, subnet in enumerate(self.vpc.select_subnets(
                subnet_group_name=_SANDBOX_SUBNETS).subnets):
            s3files.CfnMountTarget(
                self, f"MountTarget{i}",
                # .ref (the ARN) on purpose: the mount target accepts it, and switching
                # to the bare id would replace the resource — which S3 Files rejects,
                # because it allows only one mount target per availability zone and
                # CloudFormation creates the replacement before deleting the original.
                file_system_id=self.file_system.ref,
                subnet_id=subnet.subnet_id,
                security_groups=[self.mount_sg.security_group_id],
            )

        # The permissions to create an Access Point and start sessions live on the agent's
        # execution role (agentcore stack). Mounting lives on the Code Interpreter's own
        # role below, because the sandbox holds that role's credentials and runs model-written
        # code: it must not carry the agent's Bedrock, Secrets, Cognito or checkpoint access.
        self.code_interpreter_role = iam.Role(
            self, "CodeInterpreterRole",
            assumed_by=iam.ServicePrincipal(
                "bedrock-agentcore.amazonaws.com",
                conditions={"StringEquals": {"aws:SourceAccount": account}}),
            description="Code Interpreter sandbox: mount this project's S3 Files file system, nothing else")
        # Both name sets: S3 Files runs on the EFS control plane, so the elasticfilesystem
        # aliases are what some mount paths check (measured: "S3 Files mount failed: access
        # denied" without them). The file system policy above already refuses any mount that
        # names no Access Point.
        self.code_interpreter_role.add_to_policy(iam.PolicyStatement(
            actions=["s3files:ClientMount", "s3files:ClientWrite", "s3files:GetAccessPoint",
                     "elasticfilesystem:ClientMount", "elasticfilesystem:ClientWrite",
                     "elasticfilesystem:DescribeMountTargets"],
            resources=[
                self.file_system.ref,
                f"{self.file_system.ref}/access-point/*",
                f"arn:aws:elasticfilesystem:{region}:{account}:file-system/"
                f"{self.file_system.attr_file_system_id}",
            ]))
        self.code_interpreter_role.add_to_policy(iam.PolicyStatement(
            actions=["polly:*", "transcribe:*"],
            resources=["*"]))

        # Code Interpreter traffic is the only S3 traffic in this VPC, and it needs only the
        # user-files bucket (the mount's backing store). Without this, any credential or
        # presigned URL the sandbox obtains reaches any bucket in the region.
        self.s3_endpoint.add_to_policy(iam.PolicyStatement(
            principals=[iam.AnyPrincipal()],
            actions=["s3:*"],
            resources=[user_files_bucket.bucket_arn, user_files_bucket.arn_for_objects("*")]))

        CfnOutput(self, "FileSystemId", value=self.file_system.attr_file_system_id)
        CfnOutput(self, "FileSystemArn", value=self.file_system.ref)
        CfnOutput(self, "CodeInterpreterRoleArn", value=self.code_interpreter_role.role_arn)
        CfnOutput(self, "SandboxSecurityGroupId",
                  value=self.sandbox_sg.security_group_id)
        CfnOutput(self, "SandboxSubnetIds", value=",".join(
            s.subnet_id for s in self.vpc.select_subnets(
                subnet_group_name=_SANDBOX_SUBNETS).subnets))
