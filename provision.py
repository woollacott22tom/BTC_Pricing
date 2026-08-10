"""
One-time AWS resource provisioning. Import these functions from the Colab
notebook (colab/01_aws_setup.ipynb), or run this file directly with AWS
credentials set in the environment.

Creates:
  - DynamoDB tables: btc_ticks, btc_windows (on-demand billing)
  - S3 bucket for model artifacts
  - IAM role + instance profile for the EC2 box (permissions scoped to just
    these resources -- not admin/full-access)

Run this with an IAM *user's* access keys that have permission to create
IAM roles/policies, DynamoDB tables, and S3 buckets -- not root, and not
long-lived keys you leave lying around longer than needed.
"""
from __future__ import annotations
import json
import time
import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
TICKS_TABLE = "btc_ticks"
WINDOWS_TABLE = "btc_windows"
ROLE_NAME = "btc-kalshi-ec2-role"
INSTANCE_PROFILE_NAME = "btc-kalshi-ec2-profile"
POLICY_NAME = "btc-kalshi-ec2-policy"


def create_dynamo_tables(region: str = REGION):
    ddb = boto3.client("dynamodb", region_name=region)

    for table_name, sort_key in [(TICKS_TABLE, "timestamp"), (WINDOWS_TABLE, None)]:
        try:
            key_schema = [{"AttributeName": "window_id", "KeyType": "HASH"}]
            attr_defs = [{"AttributeName": "window_id", "AttributeType": "S"}]
            if sort_key:
                key_schema.append({"AttributeName": sort_key, "KeyType": "RANGE"})
                attr_defs.append({"AttributeName": sort_key, "AttributeType": "N"})

            ddb.create_table(
                TableName=table_name,
                KeySchema=key_schema,
                AttributeDefinitions=attr_defs,
                BillingMode="PAY_PER_REQUEST",  # on-demand, no capacity planning, free-tier friendly at low volume
            )
            print(f"Creating table {table_name}...")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceInUseException":
                print(f"Table {table_name} already exists, skipping.")
            else:
                raise

    for table_name in [TICKS_TABLE, WINDOWS_TABLE]:
        waiter = ddb.get_waiter("table_exists")
        waiter.wait(TableName=table_name)
        print(f"Table {table_name} is ACTIVE.")


def create_s3_bucket(bucket_name: str, region: str = REGION):
    s3 = boto3.client("s3", region_name=region)
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        print(f"Created bucket {bucket_name}")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"Bucket {bucket_name} already exists, skipping.")
        else:
            raise

    # Block all public access -- this bucket only holds model artifacts, never expose it
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
        },
    )
    print(f"Public access blocked on {bucket_name}")


def create_ec2_role_and_profile(bucket_name: str, region: str = REGION):
    """Scoped-down role: DynamoDB read/write on just our 2 tables, S3
    read/write on just our artifacts bucket. No admin access."""
    iam = boto3.client("iam", region_name=region)
    account_id = boto3.client("sts", region_name=region).get_caller_identity()["Account"]

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }

    permissions_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Query",
                    "dynamodb:Scan", "dynamodb:UpdateItem",
                ],
                "Resource": [
                    f"arn:aws:dynamodb:{region}:{account_id}:table/{TICKS_TABLE}",
                    f"arn:aws:dynamodb:{region}:{account_id}:table/{WINDOWS_TABLE}",
                ],
            },
            {
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*",
                ],
            },
        ],
    }

    try:
        iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Scoped role for BTC Kalshi predictor EC2 ingestion/serving box",
        )
        print(f"Created role {ROLE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            print(f"Role {ROLE_NAME} already exists, skipping creation.")
        else:
            raise

    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName=POLICY_NAME,
        PolicyDocument=json.dumps(permissions_policy),
    )
    print(f"Attached inline policy {POLICY_NAME} to {ROLE_NAME}")

    try:
        iam.create_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME)
        print(f"Created instance profile {INSTANCE_PROFILE_NAME}")
        time.sleep(5)  # IAM propagation delay before role can be attached
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            print(f"Instance profile {INSTANCE_PROFILE_NAME} already exists, skipping.")
        else:
            raise

    try:
        iam.add_role_to_instance_profile(
            InstanceProfileName=INSTANCE_PROFILE_NAME, RoleName=ROLE_NAME,
        )
        print(f"Attached role {ROLE_NAME} to instance profile {INSTANCE_PROFILE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "LimitExceeded":
            print("Role already attached to instance profile, skipping.")
        else:
            raise


def provision_all(bucket_name: str, region: str = REGION):
    print("== Creating DynamoDB tables ==")
    create_dynamo_tables(region)
    print("\n== Creating S3 bucket ==")
    create_s3_bucket(bucket_name, region)
    print("\n== Creating IAM role + instance profile ==")
    create_ec2_role_and_profile(bucket_name, region)
    print(f"\nDone. When launching your EC2 instance, attach IAM instance "
          f"profile: {INSTANCE_PROFILE_NAME}")


if __name__ == "__main__":
    import sys
    bucket = sys.argv[1] if len(sys.argv) > 1 else "btc-kalshi-model-artifacts"
    provision_all(bucket)
