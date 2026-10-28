"""The VPC the agent Runtime runs in when AGENT_NETWORK=vpc and no subnets of your own are given.

Only the agent moves: the MCP Runtimes, A2A, the Lambdas and the browser keep their network.
The Code Interpreter shares this VPC when FILES_STORAGE=true, in subnets and behind an S3
endpoint of its own; nothing here accepts a connection, so model-written code there cannot
reach the agent. See docs/architecture.md.

  agent    private, routed to the NAT — the agent's fixed outbound IP
  sandbox  private; routed to the NAT only with nat_gateway=true. Created whether or not
           FILES_STORAGE is on: subnets cost nothing, and a group that came and went with
           the flag would drop exports the sandbox stack still imports.
  public   holds the NAT
"""
from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    RemovalPolicy,
    Stack,
    aws_ec2 as ec2,
    aws_logs as logs,
)
from constructs import Construct

AGENT_SUBNETS = "agent"
SANDBOX_SUBNETS = "sandbox"


class NetworkStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *,
                 availability_zones: list[str], nat_gateway: bool = False,
                 eip_allocation_id: str = "",
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        groups = [
            ec2.SubnetConfiguration(name=AGENT_SUBNETS, cidr_mask=24,
                                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            ec2.SubnetConfiguration(name="public", cidr_mask=24,
                                    subnet_type=ec2.SubnetType.PUBLIC),
            ec2.SubnetConfiguration(name=SANDBOX_SUBNETS, cidr_mask=24,
                                    subnet_type=(ec2.SubnetType.PRIVATE_WITH_EGRESS if nat_gateway
                                                 else ec2.SubnetType.PRIVATE_ISOLATED)),
        ]
        # AZs by name, resolved from AgentCore's supported AZ IDs at the first deploy and kept
        # in state: names map to different IDs per account, and re-resolving would move subnets.
        # One NAT, not one per AZ: a single fixed IP and a single hourly charge.
        self.vpc = ec2.Vpc(
            self, "Vpc",
            availability_zones=availability_zones,
            ip_addresses=ec2.IpAddresses.cidr("10.30.0.0/16"),
            nat_gateways=1,
            nat_gateway_provider=(ec2.NatProvider.gateway(eip_allocation_ids=[eip_allocation_id])
                                  if eip_allocation_id else None),
            subnet_configuration=groups,
        )
        # The agent's own S3 path, on its route tables only: since 2026-05-05 a new Runtime has
        # no service-managed S3 path, and its image layers come from S3. The sandbox gets a
        # separate endpoint, restricted to the user-files bucket, in the sandbox stack.
        agent_only = [ec2.SubnetSelection(subnet_group_name=AGENT_SUBNETS)]
        self.vpc.add_gateway_endpoint("AgentS3", service=ec2.GatewayVpcEndpointAwsService.S3,
                                      subnets=agent_only)
        self.vpc.add_gateway_endpoint("AgentDynamoDb",
                                      service=ec2.GatewayVpcEndpointAwsService.DYNAMODB,
                                      subnets=agent_only)

        # Egress-only: nothing in this VPC listens, and everything the agent calls is HTTPS.
        # NTP too: the microVM syncs its clock against public servers (seen in the flow logs,
        # rejected without this), and SigV4 and JWT checks need the time right.
        self.agent_sg = ec2.SecurityGroup(
            self, "AgentSg", vpc=self.vpc, allow_all_outbound=False,
            description="Agent Runtime: HTTPS out, nothing in")
        self.agent_sg.add_egress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(443), "HTTPS out")
        self.agent_sg.add_egress_rule(ec2.Peer.any_ipv4(), ec2.Port.udp(123), "NTP out")

        # Every connection in or out, at IP level: the record a compliance review asks for.
        self.vpc.add_flow_log("FlowLog", destination=ec2.FlowLogDestination.to_cloud_watch_logs(
            logs.LogGroup(self, "FlowLogs", retention=logs.RetentionDays.ONE_MONTH,
                          removal_policy=RemovalPolicy.DESTROY)))

        CfnOutput(self, "AgentSubnetIds", value=",".join(
            s.subnet_id for s in self.vpc.select_subnets(subnet_group_name=AGENT_SUBNETS).subnets))
        CfnOutput(self, "AgentSecurityGroupId", value=self.agent_sg.security_group_id)
        # The address a downstream allowlists. A brought EIP is the operator's own to look up.
        eips = [c for c in self.vpc.node.find_all() if isinstance(c, ec2.CfnEIP)]
        if eips:
            CfnOutput(self, "NatPublicIp", value=eips[0].ref)
