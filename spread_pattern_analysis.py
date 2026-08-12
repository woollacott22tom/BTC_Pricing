"""
Tests a classic market-microstructure hypothesis directly against your own
collected data: does the bid-ask spread WIDEN before periods of higher
volatility? (Market makers often pull back / widen quotes ahead of
anticipated moves, so spread behavior can be a leading indicator even
before price itself has moved much.)

Unlike the Kalshi lead-lag/convergence analyses, this only needs Coinbase
tick data (already being collected continuously) -- runnable immediately,
no waiting period required.

Method:
  1. Pull all ticks for each closed window (or a recent slice of windows).
  2. Compute spread(t) and realized volatility over the FOLLOWING N seconds,
     for a range of N.
  3. Correlate spread(t) [and spread z-score(t)] against future volatility.
     A meaningfully positive correlation supports "wide spread predicts
     upcoming volatility."

Usage:
    python3 spread_pattern_analysis.py [--windows 100] [--horizons 5,10,15,20,30]
"""
from __future__ import annotations
import argparse
import os
import sys
import logging

import numpy as np
import pandas as pd
import boto3
from boto3.dynamodb.conditions import Key

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("spread_pattern")

REGION = os.environ.get("AWS_REGION", "us-east-1")
TICKS_TABLE = "btc_ticks"


def get_recent_window_ids(limit: int, region: str = REGION) -> list[str]:
    """btc_windows is the authoritative list of closed windows; reuse it to
    find which windows have tick data worth analyzing."""
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table("btc_windows")
    items = []
    scan_kwargs = {"ProjectionExpression": "window_id, closed_at"}
    while True:
        resp = table.scan(**scan_kwargs)
        items.extend(resp["Items"])
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    # prefer live-collected windows (which have tick data) over backfilled
    # ones (which don't) -- sort by closed_at descending, most recent first
    items = [i for i in items if "closed_at" in i]
    items.sort(key=lambda x: float(x["closed_at"]), reverse=True)
    return [i["window_id"] for i in items[:limit]]


def get_window_ticks(window_id: str, region: str = REGION) -> list[dict]:
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(TICKS_TABLE)
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


def build_spread_vol_series(ticks: list[dict]) -> pd.DataFrame | None:
    rows = []
    for t in ticks:
        try:
            best_bid = float(t["best_bid"])
            best_ask = float(t["best_ask"])
            price = float(t["price"])
            ts = float(t["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({"timestamp": ts, "spread": best_ask - best_bid, "price": price})

    if len(rows) < 20:
        return None
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def future_realized_vol(df: pd.DataFrame, idx: int, horizon_seconds: float) -> float | None:
    """Realized vol (std of log returns) over the horizon_seconds AFTER
    the observation at idx."""
    t0 = df["timestamp"].iloc[idx]
    future = df[(df["timestamp"] > t0) & (df["timestamp"] <= t0 + horizon_seconds)]
    if len(future) < 3:
        return None
    prices = future["price"].values
    prices = prices[prices > 0]
    if len(prices) < 3:
        return None
    log_ret = np.diff(np.log(prices))
    return float(np.std(log_ret))


def spread_vol_correlation(df: pd.DataFrame, horizon_seconds: float, sample_every: int = 3) -> tuple[float | None, int]:
    """Correlates spread(t) against realized vol over the following
    horizon_seconds. Subsamples (sample_every) to keep this fast across
    many windows -- doesn't need every single tick, just enough points."""
    spreads = []
    future_vols = []
    for i in range(0, len(df) - 1, sample_every):
        fv = future_realized_vol(df, i, horizon_seconds)
        if fv is not None:
            spreads.append(df["spread"].iloc[i])
            future_vols.append(fv)

    if len(spreads) < 10:
        return None, len(spreads)

    spreads_arr = np.array(spreads)
    vols_arr = np.array(future_vols)
    if spreads_arr.std() == 0 or vols_arr.std() == 0:
        return None, len(spreads)
    return float(np.corrcoef(spreads_arr, vols_arr)[0, 1]), len(spreads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=100, help="How many recent windows to analyze")
    parser.add_argument("--horizons", type=str, default="5,10,15,20,30")
    parser.add_argument("--region", type=str, default=REGION)
    args = parser.parse_args()

    horizon_list = [float(h) for h in args.horizons.split(",")]

    window_ids = get_recent_window_ids(args.windows, region=args.region)
    log.info(f"Analyzing {len(window_ids)} recent windows")

    if not window_ids:
        log.warning("No windows found. Let ingestion run longer.")
        sys.exit(0)

    results: dict[float, list[float]] = {h: [] for h in horizon_list}
    sample_counts: dict[float, int] = {h: 0 for h in horizon_list}
    windows_used = 0

    for wid in window_ids:
        ticks = get_window_ticks(wid, region=args.region)
        df = build_spread_vol_series(ticks)
        if df is None:
            continue

        used_this_window = False
        for h in horizon_list:
            corr, n = spread_vol_correlation(df, h)
            if corr is not None:
                results[h].append(corr)
                sample_counts[h] += n
                used_this_window = True

        if used_this_window:
            windows_used += 1

    log.info(f"Used {windows_used} windows with sufficient tick data")
    if windows_used == 0:
        log.warning("No windows had enough tick data yet. Let ingestion run longer.")
        sys.exit(0)

    print("\nSpread(t) -> future realized volatility correlation, by horizon:")
    print("(positive correlation = wider spread now predicts higher volatility ahead)")
    for h in horizon_list:
        corrs = results[h]
        if not corrs:
            continue
        avg = float(np.mean(corrs))
        bar = "#" * int(abs(avg) * 50)
        print(f"  {h:5.0f}s (n={len(corrs):3d} windows, ~{sample_counts[h]:5d} samples) : {avg:+.4f}  {bar}")

    print(f"\nCaveat: as with the other analyses, judge correlation STRENGTH, not just "
          f"which horizon 'wins'. Values under ~0.2-0.3 suggest weak/inconclusive evidence "
          f"even if positive.")


if __name__ == "__main__":
    main()
