"""
Empirically tests whether Coinbase price moves lead Kalshi's YES-contract
price, and by how many seconds -- rather than trusting the visual
impression. Cross-correlates Coinbase price returns against Kalshi mid-price
returns at a range of time lags; the lag with the highest correlation is
the estimated lead/lag relationship.

Method, per overlapping window:
  1. Pull Coinbase ticks (btc_ticks) and Kalshi price polls (kalshi_prices)
     for that window.
  2. Align them onto a common timeline via an as-of merge -- for each
     Kalshi observation, attach the most recent Coinbase price at or
     before that moment (pandas merge_asof, direction='backward').
  3. Compute log returns for both series on this aligned grid.
  4. For each candidate lag (in units of ~2s, matching the Kalshi poll
     interval), correlate coinbase_return(t) against kalshi_return(t+lag).
     A positive lag with the highest correlation means Coinbase's move at
     time t predicts Kalshi's move `lag` seconds later -- i.e. Coinbase
     leads.
  5. Correlations are computed PER WINDOW (never across a window
     boundary, which would create false correlation from resetting
     strikes) and then averaged across all overlapping windows.

Usage:
    python3 lead_lag_analysis.py [--max-lag-seconds 20] [--poll-interval 2]
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
log = logging.getLogger("lead_lag")

REGION = os.environ.get("AWS_REGION", "us-east-1")
TICKS_TABLE = "btc_ticks"
KALSHI_TABLE = "kalshi_prices"


def get_kalshi_window_ids(region: str = REGION) -> list[str]:
    """kalshi_prices is small (one process, a few weeks max) -- safe to
    scan for distinct window_ids rather than needing a GSI."""
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(KALSHI_TABLE)
    window_ids = set()
    scan_kwargs = {"ProjectionExpression": "window_id"}
    while True:
        resp = table.scan(**scan_kwargs)
        for item in resp["Items"]:
            window_ids.add(item["window_id"])
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return sorted(window_ids)


def get_window_series(table_name: str, window_id: str, region: str = REGION) -> list[dict]:
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


def build_aligned_returns(coinbase_items: list[dict], kalshi_items: list[dict]) -> pd.DataFrame | None:
    """Returns a DataFrame with columns [timestamp, coinbase_return,
    kalshi_return] aligned via as-of merge, or None if either side is too
    sparse to be useful."""
    if len(coinbase_items) < 5 or len(kalshi_items) < 5:
        return None

    cb_df = pd.DataFrame({
        "timestamp": [float(t["timestamp"]) for t in coinbase_items],
        "price": [float(t["price"]) for t in coinbase_items],
    }).sort_values("timestamp")

    kl_df = pd.DataFrame({
        "timestamp": [float(t["timestamp"]) for t in kalshi_items],
        "yes_mid_cents": [float(t["yes_mid_cents"]) for t in kalshi_items],
    }).sort_values("timestamp")

    merged = pd.merge_asof(
        kl_df, cb_df, on="timestamp", direction="backward", tolerance=10.0
    )
    merged = merged.dropna(subset=["price", "yes_mid_cents"])
    if len(merged) < 5:
        return None

    merged["coinbase_return"] = np.log(merged["price"]).diff()
    merged["kalshi_return"] = merged["yes_mid_cents"].diff() / 100.0  # keep units comparable-ish (small deltas)
    merged = merged.dropna(subset=["coinbase_return", "kalshi_return"])

    return merged[["timestamp", "coinbase_return", "kalshi_return"]]


def correlation_at_lags(df: pd.DataFrame, max_lag_steps: int) -> dict[int, float]:
    """Positive lag_steps: coinbase_return(t) correlated against
    kalshi_return(t + lag_steps) -- i.e. does coinbase's move predict
    kalshi's move `lag_steps` observations later."""
    results = {}
    n = len(df)
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag >= 0:
            a = df["coinbase_return"].iloc[: n - lag].reset_index(drop=True)
            b = df["kalshi_return"].iloc[lag:].reset_index(drop=True)
        else:
            a = df["coinbase_return"].iloc[-lag:].reset_index(drop=True)
            b = df["kalshi_return"].iloc[: n + lag].reset_index(drop=True)

        if len(a) < 5 or a.std() == 0 or b.std() == 0:
            continue
        results[lag] = float(np.corrcoef(a, b)[0, 1])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lag-seconds", type=int, default=20)
    parser.add_argument("--poll-interval", type=float, default=2.0,
                         help="Kalshi poll interval in seconds -- sets the lag step size")
    parser.add_argument("--region", type=str, default=REGION)
    args = parser.parse_args()

    max_lag_steps = int(args.max_lag_seconds / args.poll_interval)

    window_ids = get_kalshi_window_ids(region=args.region)
    log.info(f"Found {len(window_ids)} windows with Kalshi data")

    if len(window_ids) == 0:
        log.warning("No Kalshi data found yet. Let the poller run longer before analyzing.")
        sys.exit(0)

    all_lag_correlations: dict[int, list[float]] = {lag: [] for lag in range(-max_lag_steps, max_lag_steps + 1)}
    windows_used = 0

    for wid in window_ids:
        coinbase_items = get_window_series(TICKS_TABLE, wid, region=args.region)
        kalshi_items = get_window_series(KALSHI_TABLE, wid, region=args.region)

        aligned = build_aligned_returns(coinbase_items, kalshi_items)
        if aligned is None:
            continue

        lag_corrs = correlation_at_lags(aligned, max_lag_steps)
        if not lag_corrs:
            continue

        for lag, corr in lag_corrs.items():
            all_lag_correlations[lag].append(corr)
        windows_used += 1

    log.info(f"Used {windows_used} windows with sufficient overlapping data")

    if windows_used == 0:
        log.warning("No windows had enough overlapping Coinbase + Kalshi data to analyze. "
                     "Let both pollers run longer.")
        sys.exit(0)

    avg_corr_by_lag = {
        lag: float(np.mean(corrs)) for lag, corrs in all_lag_correlations.items() if corrs
    }

    print("\nLag (seconds) -> average correlation across windows:")
    print("(positive lag = Coinbase move precedes Kalshi move by that many seconds)")
    for lag_steps in sorted(avg_corr_by_lag.keys()):
        lag_seconds = lag_steps * args.poll_interval
        bar = "#" * int(abs(avg_corr_by_lag[lag_steps]) * 50)
        print(f"  {lag_seconds:+6.1f}s : {avg_corr_by_lag[lag_steps]:+.4f}  {bar}")

    best_lag_steps = max(avg_corr_by_lag, key=lambda k: avg_corr_by_lag[k])
    best_lag_seconds = best_lag_steps * args.poll_interval
    best_corr = avg_corr_by_lag[best_lag_steps]

    print(f"\nStrongest correlation at lag = {best_lag_seconds:+.1f}s (correlation = {best_corr:+.4f})")
    if best_lag_steps > 0:
        print(f"Interpretation: Coinbase price moves appear to precede Kalshi's price "
              f"by roughly {best_lag_seconds:.1f} seconds, on average across {windows_used} windows.")
    elif best_lag_steps < 0:
        print(f"Interpretation: Kalshi's price appears to move BEFORE Coinbase's, by roughly "
              f"{abs(best_lag_seconds):.1f} seconds -- opposite of the front-running hypothesis.")
    else:
        print("Interpretation: no meaningful lead/lag detected -- prices move together with no offset.")

    print(f"\nCaveat: correlation strength ({best_corr:+.4f}) matters as much as the lag itself. "
          f"A weak correlation (well under ~0.2-0.3) even at the 'best' lag suggests this "
          f"relationship may be noisy or not reliably tradeable, regardless of direction.")


if __name__ == "__main__":
    main()
