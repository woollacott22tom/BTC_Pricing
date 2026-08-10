"""
Builds short-lived JWTs for Coinbase Advanced Trade's WebSocket API, using
an Ed25519 CDP (Cloud Developer Platform) API key stored in AWS Secrets
Manager.

IMPORTANT -- Coinbase's Ed25519 keys are NOT given as a PEM file. The
"copy" button in the CDP dashboard gives you a raw base64-encoded string
that decodes to exactly 64 bytes: a 32-byte private seed followed by a
32-byte public key (the standard libsodium/NaCl Ed25519 secret key
format). This is different from Coinbase's older ECDSA keys, which ARE
PEM-wrapped and use ES256 instead of EdDSA.

The secret in Secrets Manager is a JSON string with two fields:
  {"api_key_name": "organizations/{org_id}/apiKeys/{key_id}", "private_key_b64": "<raw base64 string from the copy button, no PEM wrapper>"}

Environment variable required:
  COINBASE_SECRET_NAME   e.g. "btc-kalshi/coinbase-api-key"
  AWS_REGION              defaults to us-east-1 if unset
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
import time

import boto3
import jwt as pyjwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

_cached_secret: dict | None = None
_cached_key_obj: Ed25519PrivateKey | None = None
_cache_fetched_at: float = 0.0
_CACHE_TTL_SEC = 300


def _load_credentials() -> tuple[str, Ed25519PrivateKey]:
    global _cached_secret, _cached_key_obj, _cache_fetched_at

    now = time.time()
    if _cached_secret is not None and (now - _cache_fetched_at) < _CACHE_TTL_SEC:
        return _cached_secret["api_key_name"], _cached_key_obj

    secret_name = os.environ.get("COINBASE_SECRET_NAME")
    if not secret_name:
        raise RuntimeError("COINBASE_SECRET_NAME must be set in the environment.")

    region = os.environ.get("AWS_REGION", "us-east-1")
    client = boto3.client("secretsmanager", region_name=region)
    resp = client.get_secret_value(SecretId=secret_name)
    secret = json.loads(resp["SecretString"])

    raw_bytes = base64.b64decode(secret["private_key_b64"])
    if len(raw_bytes) != 64:
        raise ValueError(
            f"Expected a 64-byte Ed25519 secret key (32-byte seed + 32-byte "
            f"public key), got {len(raw_bytes)} bytes. Check that "
            f"private_key_b64 is the raw string Coinbase's copy button gave "
            f"you, with no PEM wrapper or extra whitespace."
        )
    seed = raw_bytes[:32]  # cryptography wants just the seed, not the full 64-byte NaCl format
    key_obj = Ed25519PrivateKey.from_private_bytes(seed)

    _cached_secret = secret
    _cached_key_obj = key_obj
    _cache_fetched_at = now
    return secret["api_key_name"], key_obj


def build_ws_jwt() -> str:
    """Returns a fresh signed JWT for use in an Advanced Trade WS subscribe message."""
    api_key, key_obj = _load_credentials()
    payload = {
        "iss": "cdp",
        "nbf": int(time.time()),
        "exp": int(time.time()) + 120,
        "sub": api_key,
    }
    headers = {
        "kid": api_key,
        "nonce": hashlib.sha256(os.urandom(16)).hexdigest(),
    }
    return pyjwt.encode(payload, key_obj, algorithm="EdDSA", headers=headers)
