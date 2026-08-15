"""
Tests a specific hypothesis: Kraken's persistent negative book imbalance
is arbitrage pressure, not a bug -- if Kraken trades at a premium to
Coinbase (thinner/less liquid market temporarily priced high), arbitrageurs
sell into that premium (buy cheap elsewhere, sell expensive on Kraken),
and that selling pressure should show up as negative book imbalance on
Kraken specifically.

Method, per window with both Coinbase and Kraken tick data:
  1. Align Coinbase's price onto Kraken's (sparser) tick timestamps via an
     as-of merge -- same technique validated in lead_lag_analysis.py.
  2. Compute premium(t) = kraken_price(t) - coinbase_price(t) at each
     Kraken tick.
  3. Correlate premium(t) against kraken_book_imbalance(t) (and each depth
     tier) at the same tick, per window, averaged across windows.

If the arbitrage-pressure theory holds: premium > 0 (Kraken pricier)
should correlate with book_imbalance < 0 (more resting asks) -- i.e. a
NEGATIVE correlation between premium and imbalance.

Usage:
    python3 kraken_premium_arbitrage_check.py [--windows 100]
"""
from __future__ import annotations
import argparse
import os
import logging

import numpy as np
import pandas as pd
import boto3
from boto3.dynamodb.conditions import Key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kraken_arb_check")

REGION = os.environ.get("AWS_REGION", "us-east-1")


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
    return sorted(items, key=lambda x: float(x["timestamp"]))


IMBALANCE_FIELDS = {
    "book_imbalance": "feat_book_imbalance",
    "imbalance_5": "feat_imbalance_5",
    "imbalance_20": "feat_imbalance_20",
    "imbalance_50": "feat_imbalance_50",
}


def build_premium_imbalance_df(cb_ticks: list[dict], kr_ticks: list[dict]) -> pd.DataFrame | None:
    if len(cb_ticks) < 3 or len(kr_ticks) < 3:
        return None

    cb_df = pd.DataFrame({
        "timestamp": [float(t["timestamp"]) for t in cb_ticks],
        "cb_price": [float(t["price"]) for t in cb_ticks],
    }).sort_values("timestamp")

    kr_rows = []
    for t in kr_ticks:
        row = {"timestamp": float(t["timestamp"]), "kr_price": float(t["price"])}
        for label, key in IMBALANCE_FIELDS.items():
            if key in t and t[key] is not None:
                row[label] = float(t[key])
        kr_rows.append(row)
    kr_df = pd.DataFrame(kr_rows).sort_values("timestamp")

    merged = pd.merge_asof(kr_df, cb_df, on="timestamp", direction="backward", tolerance=10.0)
    merged = merged.dropna(subset=["cb_price", "kr_price"])
    if len(merged) < 5:
        return None

    merged["premium"] = merged["kr_price"] - merged["cb_price"]
    return merged


def correlate_premium_vs_imbalance(df: pd.DataFrame) -> dict[str, float | None]:
    results = {}
    for label in IMBALANCE_FIELDS:
        if label not in df.columns:
            results[label] = None
            continue
        sub = df.dropna(subset=["premium", label])
        if len(sub) < 5 or sub["premium"].std() == 0 or sub[label].std() == 0:
            results[label] = None
            continue
        results[label] = float(np.corrcoef(sub["premium"], sub[label])[0, 1])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--region", type=str, default=REGION)
    args = parser.parse_args()

    window_ids = get_recent_window_ids(args.windows, region=args.region)
    log.info(f"Checking {len(window_ids)} recent windows")

    all_corrs: dict[str, list[float]] = {label: [] for label in IMBALANCE_FIELDS}
    all_premiums = []
    windows_used = 0

    for wid in window_ids:
        cb_ticks = get_window_ticks("btc_ticks", wid, region=args.region)
        kr_ticks = get_window_ticks("kraken_ticks", wid, region=args.region)

        df = build_premium_imbalance_df(cb_ticks, kr_ticks)
        if df is None:
            continue

        all_premiums.extend(df["premium"].tolist())
        corrs = correlate_premium_vs_imbalance(df)
        used_this_window = False
        for label, c in corrs.items():
            if c is not None:
                all_corrs[label].append(c)
                used_this_window = True
        if used_this_window:
            windows_used += 1

    log.info(f"Used {windows_used} windows with sufficient data\n")

    if all_premiums:
        premiums = np.array(all_premiums)
        print(f"Kraken premium over Coinbase: mean=${premiums.mean():+.2f}  "
              f"median=${np.median(premiums):+.2f}  "
              f"%time_premium_positive={float(np.mean(premiums > 0))*100:.1f}%")
        print()

    print("Premium(t) vs book_imbalance(t) correlation, by depth tier:")
    print("(theory predicts NEGATIVE correlation: premium up -> imbalance down,")
    print(" i.e. arb sellers building ask-side pressure while Kraken is priced high)")
    for label, corrs in all_corrs.items():
        if not corrs:
            print(f"  {label}: insufficient data")
            continue
        avg = float(np.mean(corrs))
        bar = "#" * int(abs(avg) * 50)
        direction = "supports arbitrage theory" if avg < -0.1 else ("CONTRADICTS theory" if avg > 0.1 else "inconclusive")
        print(f"  {label} (n={len(corrs)} windows): {avg:+.4f}  {bar}  [{direction}]")

    print(f"\nCaveat: correlation strength matters as much as sign. Treat |correlation|")
    print(f"under ~0.2-0.3 as weak/inconclusive evidence either way.")


if __name__ == "__main__":
    main()
