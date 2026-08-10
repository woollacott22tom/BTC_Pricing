"""
Thin DynamoDB wrapper. Two tables:

  btc_ticks
    PK: window_id (S)   e.g. "2026-08-10T14:15:00+00:00"
    SK: timestamp (N)   unix seconds, float
    attrs: price, volume, best_bid, best_ask, bid_depth_top10, ask_depth_top10,
           + the computed feature snapshot (flattened)

  btc_windows   (one item per closed window -- the training label table)
    PK: window_id (S)
    attrs: open_price (strike), settlement_price (60s TWAP), outcome ("up"/"down"),
           max_deviation_final_60s, flip_occurred (bool), closed_at

Both tables use on-demand billing to stay simple and within free tier at this
data volume (a handful of writes/sec, not sustained high throughput).
"""
from __future__ import annotations
import boto3
from decimal import Decimal
from boto3.dynamodb.conditions import Key

TICKS_TABLE = "btc_ticks"
WINDOWS_TABLE = "btc_windows"

_dynamodb = None


def get_resource(region_name: str = "us-east-1"):
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource("dynamodb", region_name=region_name)
    return _dynamodb


def _to_decimal(obj):
    """DynamoDB requires Decimal, not float. Recursively convert, dropping
    None values (DynamoDB has no null-friendly numeric type for our use)."""
    if isinstance(obj, dict):
        return {k: _to_decimal(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, list):
        return [_to_decimal(v) for v in obj]
    return obj


def put_tick(window_id: str, timestamp: float, tick_fields: dict, features: dict, region_name: str = "us-east-1"):
    table = get_resource(region_name).Table(TICKS_TABLE)
    item = {"window_id": window_id, "timestamp": Decimal(str(timestamp))}
    item.update(_to_decimal(tick_fields))
    item.update({f"feat_{k}": v for k, v in _to_decimal(features).items()})
    table.put_item(Item=item)


def put_window_summary(window_id: str, summary: dict, region_name: str = "us-east-1"):
    table = get_resource(region_name).Table(WINDOWS_TABLE)
    item = {"window_id": window_id}
    item.update(_to_decimal(summary))
    table.put_item(Item=item)


def get_window_ticks(window_id: str, region_name: str = "us-east-1") -> list[dict]:
    table = get_resource(region_name).Table(TICKS_TABLE)
    items = []
    resp = table.query(KeyConditionExpression=Key("window_id").eq(window_id))
    items.extend(resp["Items"])
    while "LastEvaluatedKey" in resp:
        resp = table.query(
            KeyConditionExpression=Key("window_id").eq(window_id),
            ExclusiveStartKey=resp["LastEvaluatedKey"],
        )
        items.extend(resp["Items"])
    return sorted(items, key=lambda x: float(x["timestamp"]))


def list_closed_windows(region_name: str = "us-east-1", limit: int | None = None) -> list[dict]:
    """Full scan of btc_windows. Fine at this scale (thousands of items max);
    revisit with a GSI on closed_at if this table grows very large."""
    table = get_resource(region_name).Table(WINDOWS_TABLE)
    items = []
    scan_kwargs = {}
    while True:
        resp = table.scan(**scan_kwargs)
        items.extend(resp["Items"])
        if limit and len(items) >= limit:
            return items[:limit]
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items
