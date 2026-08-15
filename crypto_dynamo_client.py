"""
DynamoDB wrapper for Crypto.com's ticks table, kept separate from the
other exchanges' clients so nothing here can risk any other pipeline.
"""
from __future__ import annotations
import boto3
from decimal import Decimal

TABLE_NAME = "cryptocom_ticks"
_dynamodb = None


def get_resource(region_name: str = "us-east-1"):
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource("dynamodb", region_name=region_name)
    return _dynamodb


def _to_decimal(obj):
    if isinstance(obj, dict):
        return {k: _to_decimal(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, list):
        return [_to_decimal(v) for v in obj]
    return obj


def put_cryptocom_tick(window_id: str, timestamp: float, tick_fields: dict, features: dict, region_name: str = "us-east-1"):
    table = get_resource(region_name).Table(TABLE_NAME)
    item = {"window_id": window_id, "timestamp": Decimal(str(timestamp))}
    item.update(_to_decimal(tick_fields))
    item.update({f"feat_{k}": v for k, v in _to_decimal(features).items()})
    table.put_item(Item=item)
