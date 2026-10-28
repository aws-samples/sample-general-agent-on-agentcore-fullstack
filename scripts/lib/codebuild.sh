# shellcheck shell=bash
# Build a container image natively on ARM64 in CodeBuild and push it to ECR.
# Sourced by build-mcp.sh and provision.sh, which set ACCOUNT, REGION and PREFIX first.
#
# Native because the Runtimes are ARM64 and a QEMU cross-build is slow and unreliable
# for native wheels; remote so a deploy needs neither Docker nor a particular CPU.

# One project per region builds every image (agent and MCP servers). The role is global
# in IAM, so it carries the region: destroying one region must not break another.
CB_PROJECT="$PREFIX-image-builder"
CB_ROLE_NAME="$PREFIX-image-builder-$REGION"
# Shared with the AgentCore CLI, which normally creates it; created here if absent.
CB_SRC_BUCKET="bedrock-agentcore-codebuild-sources-$ACCOUNT-$REGION"

_cb_ready=""

cb_ensure() { # role, source bucket and project — once per process
  [ -n "$_cb_ready" ] && return 0
  if ! aws iam get-role --role-name "$CB_ROLE_NAME" >/dev/null 2>&1; then
    aws iam create-role --role-name "$CB_ROLE_NAME" \
      --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"codebuild.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  fi
  # ECR push by prefix: every image this project builds lives in a $PREFIX-* repo.
  aws iam put-role-policy --role-name "$CB_ROLE_NAME" --policy-name build \
    --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[
      {\"Effect\":\"Allow\",\"Action\":[\"logs:CreateLogGroup\",\"logs:CreateLogStream\",\"logs:PutLogEvents\"],\"Resource\":\"arn:aws:logs:$REGION:$ACCOUNT:log-group:/aws/codebuild/$CB_PROJECT*\"},
      {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:GetObjectVersion\"],\"Resource\":\"arn:aws:s3:::$CB_SRC_BUCKET/$PREFIX-*\"},
      {\"Effect\":\"Allow\",\"Action\":\"ecr:GetAuthorizationToken\",\"Resource\":\"*\"},
      {\"Effect\":\"Allow\",\"Action\":[\"ecr:BatchCheckLayerAvailability\",\"ecr:PutImage\",\"ecr:InitiateLayerUpload\",\"ecr:UploadLayerPart\",\"ecr:CompleteLayerUpload\",\"ecr:BatchGetImage\",\"ecr:GetDownloadUrlForLayer\"],\"Resource\":\"arn:aws:ecr:$REGION:$ACCOUNT:repository/$PREFIX-*\"}]}" >/dev/null
  local role_arn proj
  sleep 8  # the role and its policy must propagate before CodeBuild validates them
  role_arn="$(aws iam get-role --role-name "$CB_ROLE_NAME" --query 'Role.Arn' --output text)"
  aws s3api head-bucket --bucket "$CB_SRC_BUCKET" >/dev/null 2>&1 || \
    aws s3api create-bucket --bucket "$CB_SRC_BUCKET" \
      --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
  # Only the environment lives on the project; source and buildspec are passed per build,
  # so one project serves every image without being rewritten between them.
  proj="$(ROLE="$role_arn" NAME="$CB_PROJECT" LOC="$CB_SRC_BUCKET/$PREFIX-placeholder/source.zip" python3 -c '
import json, os
e = os.environ
print(json.dumps({
    "name": e["NAME"],
    "source": {"type": "S3", "location": e["LOC"], "buildspec": "version: 0.2\nphases:\n  build:\n    commands:\n      - exit 1\n"},
    "artifacts": {"type": "NO_ARTIFACTS"},
    "environment": {"type": "ARM_CONTAINER", "image": "aws/codebuild/amazonlinux2-aarch64-standard:3.0",
                    "computeType": "BUILD_GENERAL1_MEDIUM", "privilegedMode": True},
    "serviceRole": e["ROLE"],
}))')"
  if aws codebuild batch-get-projects --names "$CB_PROJECT" --query 'projects[0].name' --output text 2>/dev/null | grep -q "$CB_PROJECT"; then
    aws codebuild update-project --cli-input-json "$proj" >/dev/null
  else
    aws codebuild create-project --cli-input-json "$proj" >/dev/null
  fi
  _cb_ready=1
}

cb_image_exists() { # repo tag
  aws ecr describe-images --repository-name "$1" --image-ids "imageTag=$2" >/dev/null 2>&1
}

cb_build() { # repo source-dir tag [docker build args...] — builds and pushes repo:tag, sets CB_IMAGE_URI
  # Call it bare, not under || or if: either would switch off set -e inside it.
  local repo="$1" dir="$2" tag="$3"; shift 3
  local ecr="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$repo" key="$repo/source.zip"
  cb_ensure
  aws ecr describe-repositories --repository-names "$repo" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "$repo" >/dev/null
  local zipf
  zipf="$(mktemp -u).zip"   # -u: name only; zip rejects a pre-existing empty file
  ( cd "$dir" && zip -qr "$zipf" . -x '*.pyc' '*__pycache__/*' -x 'test_*' )
  aws s3 cp "$zipf" "s3://$CB_SRC_BUCKET/$key" >/dev/null
  rm -f "$zipf"  # safe-rm-ok
  local spec
  spec="$(ECR="$ecr" TAG="$tag" ARGS="$*" python3 -c '
import os
e = os.environ
ref = e["ECR"] + ":" + e["TAG"]
print("\n".join([
    "version: 0.2",
    "phases:",
    "  pre_build:",
    "    commands:",
    "      - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin " + e["ECR"],
    "  build:",
    "    commands:",
    "      - docker build " + e["ARGS"] + " -t " + ref + " .",
    "  post_build:",
    "    commands:",
    "      - docker push " + ref,
]))')"
  local bid phase="" last="" status
  bid="$(aws codebuild start-build --project-name "$CB_PROJECT" \
    --source-type-override S3 --source-location-override "$CB_SRC_BUCKET/$key" \
    --buildspec-override "$spec" --query 'build.id' --output text)"
  echo "  CodeBuild $bid"
  while :; do
    read -r status phase < <(aws codebuild batch-get-builds --ids "$bid" \
      --query 'builds[0].[buildStatus,currentPhase]' --output text)
    [ "$phase" != "$last" ] && { echo "    $phase"; last="$phase"; }
    [ "$status" = "IN_PROGRESS" ] || break
    sleep 10
  done
  if [ "$status" != "SUCCEEDED" ]; then
    echo "  build $status — logs: https://$REGION.console.aws.amazon.com/codesuite/codebuild/projects/$CB_PROJECT/build/${bid//:/%3A}/log"
    return 1
  fi
  # shellcheck disable=SC2034  # read by the caller
  CB_IMAGE_URI="$ecr:$tag"
}
