"""
Checks whether each exchange's book_imbalance (and tier imbalances) shows
a genuine, persistent skew -- specifically testing the observation that
Kraken's book has looked negative "the whole time" on the dashboard.

A persistent skew could mean two very different things:
  1. Real market structure -- Kraken's BTC/USD book genuinely tends to
     carry more resting ask-side size than bid-side for structural reasons
     (e.g. market maker positioning specific to that venue). This would
     be a real, if unhelpful, feature -- a near-constant value carries
     little predictive signal but isn't WRONG.
  2. A bug in kraken_order_book.py's bid/ask parsing -- e.g. a
     side-labeling mix-up that systematically inflates one side. This
     would mean the feature is actively misleading, not just uninformative,
     and needs fixing before it's trusted in training.

This script only reports the numbers -- it doesn't diagnose which case
applies, since that requires human judgment about what's plausible for
this specific market. Compare the reported means: a swing that's an order
of magnitude different from Coinbase/Crypto.com's, or a near-100%-negative
rate, points more toward case 2; a modest, consistent skew shared in
direction (even if not magnitude) with the other exchanges points more
toward case 1.

Usage:
    python3 imbalance_consensus_check.py [--windows 100]
"""
from __future__ import annotations
import argparse
import os
import logging

import numpy as np
import boto3
from boto3.dynamodb.conditions import Key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("imbalance_check")

REGION = os.environ.get("AWS_REGION", "us-east-1")
TABLES = {"Coinbase": "btc_ticks", "Kraken": "kraken_ticks", "Crypto.com": "cryptocom_ticks"}


def get_recent_window_ids(limit: int, region: str = REGION) -> list[str]:
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table("btc_windows")
    items = []
    scan_kwargs = {"ProjectionExpression": "window_id, closed_at, #src", "ExpressionAttributeNames": {"#src": "source"}}
    while True:
        resp = table.scan(**scan_kwargs)
        items.extend(resp["Items"])
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    items = [i for i in items if "closed_at" in i and i.get("source") != "backfill"]
    items.sort(key=lambda x: float(x["closed_at"]), reverse=True)
    return [i["window_id"] for i in items[:limit]]


def get_window_ticks(table_name: str, window_id: str, region: str = REGION) -> list[dict]:
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(table_name)
    items = []
    resp = table.query(KeyConditionExpression=Key("window_id").eq(window_id))
    items.extend(resp["Items"])
    while "LastEvaluatedKey" in resp:
        resp = table.query(
            KeyConditionExpression=Key("window_id").eq(window_id),
            ExclusiveStartKey=resp["LastEvaluatedKey"],
        )
        items.extend(resp["Items"])
    return items


def summarize_field(values: list[float], label: str):
    if not values:
        print(f"  {label}: no data")
        return
    arr = np.array(values, dtype=float)
    pct_negative = float(np.mean(arr < 0)) * 100
    pct_positive = float(np.mean(arr > 0)) * 100
    print(f"  {label}: n={len(arr)}  mean={arr.mean():+.4f}  median={np.median(arr):+.4f}  "
          f"std={arr.std():.4f}  min={arr.min():+.4f}  max={arr.max():+.4f}  "
          f"%negative={pct_negative:.1f}%  %positive={pct_positive:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--region", type=str, default=REGION)
    args = parser.parse_args()

    window_ids = get_recent_window_ids(args.windows, region=args.region)
    log.info(f"Checking {len(window_ids)} recent windows across all three exchanges\n")

    fields_to_check = ["feat_book_imbalance", "feat_imbalance_5", "feat_imbalance_20", "feat_imbalance_50"]

    for exchange_name, table_name in TABLES.items():
        print(f"=== {exchange_name} ({table_name}) ===")
        field_values = {f: [] for f in fields_to_check}

        for wid in window_ids:
            ticks = get_window_ticks(table_name, wid, region=args.region)
            for t in ticks:
                for f in fields_to_check:
                    if f in t and t[f] is not None:
                        field_values[f].append(float(t[f]))

        for f in fields_to_check:
            summarize_field(field_values[f], f.replace("feat_", ""))
        print()

    print("Interpretation guide:")
    print("  - If one exchange's %negative is near 90-100% while others hover closer to 50%,")
    print("    that's a strong signal of either genuine structural skew or a parsing bug --")
    print("    worth checking kraken_order_book.py's bid/ask side assignment specifically")
    print("    if Kraken is the outlier, since that's the newest/least-scrutinized code path.")
    print("  - A modest, shared-direction skew across ALL exchanges is more likely real market")
    print("    structure (e.g. broad ask-heavy conditions during a specific period) than a bug.")


if __name__ == "__main__":
    main()
