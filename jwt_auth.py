"""
Builds short-lived JWTs for Coinbase Advanced Trade's WebSocket API, using a
CDP (Cloud Developer Platform) API key stored in AWS Secrets Manager -- not
a local file, not an environment variable, and never committed to git.

The secret is a JSON string with two fields:
  {"api_key_name": "organizations/{org_id}/apiKeys/{key_id}", "private_key_pem": "-----BEGIN EC PRIVATE KEY-----..."}

Access is controlled by the EC2 instance's IAM role (same pattern as the
DynamoDB/S3 access already set up), so no credentials of any kind need to
live on disk or pass through Colab after the secret is created once.

Environment variable required:
  COINBASE_SECRET_NAME   e.g. "btc-kalshi/coinbase-api-key" (not sensitive --
                          it's just the secret's name/identifier, not its value)
  AWS_REGION              defaults to us-east-1 if unset
"""
from __future__ import annotations
import hashlib
import json
import os
import time

import boto3
import jwt as pyjwt

ALGORITHM = "ES256"

_cached_secret: dict | None = None
_cache_fetched_at: float = 0.0
_CACHE_TTL_SEC = 300  # re-fetch at most every 5 minutes; the key itself doesn't rotate on its own


def _load_credentials() -> tuple[str, str]:
    global _cached_secret, _cache_fetched_at

    now = time.time()
    if _cached_secret is not None and (now - _cache_fetched_at) < _CACHE_TTL_SEC:
        return _cached_secret["api_key_name"], _cached_secret["private_key_pem"]

    secret_name = os.environ.get("COINBASE_SECRET_NAME")
    if not secret_name:
        raise RuntimeError("COINBASE_SECRET_NAME must be set in the environment.")

    region = os.environ.get("AWS_REGION", "us-east-1")
    client = boto3.client("secretsmanager", region_name=region)
    resp = client.get_secret_value(SecretId=secret_name)
    secret = json.loads(resp["SecretString"])

    _cached_secret = secret
    _cache_fetched_at = now
    return secret["api_key_name"], secret["private_key_pem"]


def build_ws_jwt() -> str:
    """Returns a fresh signed JWT for use in an Advanced Trade WS subscribe message."""
    api_key, private_key_pem = _load_credentials()
    payload = {
        "iss": "coinbase-cloud",
        "nbf": int(time.time()),
        "exp": int(time.time()) + 120,
        "sub": api_key,
    }
    headers = {
        "kid": api_key,
        "nonce": hashlib.sha256(os.urandom(16)).hexdigest(),
    }
    return pyjwt.encode(payload, private_key_pem, algorithm=ALGORITHM, headers=headers)
