"""
Tests a different, more specific hypothesis than simple lead-lag
correlation: does the GAP between a Coinbase-implied fair-value probability
and Kalshi's current price predict Kalshi's SUBSEQUENT price movement
toward closing that gap -- i.e. convergence, not just correlated returns.

This matches a "Kalshi initially lags/overshoots, then catches up to follow
the real move" pattern better than a single fixed-lag cross-correlation
would, since it doesn't assume a constant lag -- it directly measures
whether the size of the current mispricing predicts the future correction.

Coinbase-implied fair value proxy: using the existing
distance_to_strike_stdevs feature (already computed and stored per tick),
under a standard random-walk assumption the probability price ends above
strike is approximately Phi(distance_to_strike_stdevs) -- the standard
normal CDF. This is a simple, principled statistical proxy, not a trained
model -- it's meant to test the hypothesis, not to be the final signal.

Method, per overlapping window:
  1. Merge Coinbase's distance_to_strike_stdevs onto Kalshi's poll
     timestamps (as-of, same alignment approach as lead_lag_analysis.py).
  2. Compute coinbase_implied_prob_cents = 100 * Phi(distance_to_strike_stdevs).
  3. gap(t) = coinbase_implied_prob_cents(t) - kalshi_yes_mid_cents(t)
  4. For a range of horizons, compute kalshi's price change over that
     horizon: future_kalshi_change(t, h) = kalshi_yes_mid_cents(t+h) - kalshi_yes_mid_cents(t)
  5. Correlate gap(t) against future_kalshi_change(t, h) for each horizon.
     A strong POSITIVE correlation means: when Coinbase implies Kalshi is
     underpriced (positive gap), Kalshi's price tends to rise afterward --
     i.e. real, measurable convergence, and the horizon with the strongest
     correlation is roughly how long that convergence takes.

Usage:
    python3 convergence_analysis.py [--horizons 2,4,6,8,10,14,20,30]
"""
from __future__ import annotations
import argparse
import os
import sys
import logging

import numpy as np
import pandas as pd
from scipy.stats import norm
import boto3
from boto3.dynamodb.conditions import Key

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("convergence")

REGION = os.environ.get("AWS_REGION", "us-east-1")
TICKS_TABLE = "btc_ticks"
KALSHI_TABLE = "kalshi_prices"


def get_kalshi_window_ids(region: str = REGION) -> list[str]:
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


def build_gap_series(coinbase_items: list[dict], kalshi_items: list[dict]) -> pd.DataFrame | None:
    if len(coinbase_items) < 5 or len(kalshi_items) < 5:
        return None

    cb_rows = []
    for t in coinbase_items:
        dist = t.get("feat_distance_to_strike_stdevs")
        if dist is None:
            continue
        cb_rows.append({"timestamp": float(t["timestamp"]), "distance_stdevs": float(dist)})
    if len(cb_rows) < 5:
        return None
    cb_df = pd.DataFrame(cb_rows).sort_values("timestamp")

    kl_df = pd.DataFrame({
        "timestamp": [float(t["timestamp"]) for t in kalshi_items],
        "yes_mid_cents": [float(t["yes_mid_cents"]) for t in kalshi_items],
    }).sort_values("timestamp")

    merged = pd.merge_asof(kl_df, cb_df, on="timestamp", direction="backward", tolerance=10.0)
    merged = merged.dropna(subset=["distance_stdevs", "yes_mid_cents"])
    if len(merged) < 10:
        return None

    merged["coinbase_implied_prob_cents"] = 100.0 * norm.cdf(merged["distance_stdevs"])
    merged["gap"] = merged["coinbase_implied_prob_cents"] - merged["yes_mid_cents"]

    return merged[["timestamp", "gap", "yes_mid_cents"]].reset_index(drop=True)


def gap_convergence_correlation(df: pd.DataFrame, horizon_steps: int) -> float | None:
    """Correlates gap(t) against kalshi's price change over the next
    horizon_steps observations. Positive correlation = convergence
    (positive gap predicts kalshi price rising to close it)."""
    n = len(df)
    if n <= horizon_steps + 5:
        return None

    gap_t = df["gap"].iloc[: n - horizon_steps].reset_index(drop=True)
    future_change = (
        df["yes_mid_cents"].iloc[horizon_steps:].reset_index(drop=True)
        - df["yes_mid_cents"].iloc[: n - horizon_steps].reset_index(drop=True)
    )

    if gap_t.std() == 0 or future_change.std() == 0:
        return None
    return float(np.corrcoef(gap_t, future_change)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=str, default="2,4,6,8,10,14,20,30",
                         help="Comma-separated horizons in seconds to test")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--region", type=str, default=REGION)
    args = parser.parse_args()

    horizon_seconds_list = [int(h) for h in args.horizons.split(",")]
    horizon_steps_list = [max(1, round(h / args.poll_interval)) for h in horizon_seconds_list]

    window_ids = get_kalshi_window_ids(region=args.region)
    log.info(f"Found {len(window_ids)} windows with Kalshi data")
    if not window_ids:
        log.warning("No Kalshi data found yet.")
        sys.exit(0)

    results: dict[int, list[float]] = {h: [] for h in horizon_seconds_list}
    windows_used = 0

    for wid in window_ids:
        coinbase_items = get_window_series(TICKS_TABLE, wid, region=args.region)
        kalshi_items = get_window_series(KALSHI_TABLE, wid, region=args.region)

        gap_df = build_gap_series(coinbase_items, kalshi_items)
        if gap_df is None:
            continue

        used_this_window = False
        for h_sec, h_steps in zip(horizon_seconds_list, horizon_steps_list):
            corr = gap_convergence_correlation(gap_df, h_steps)
            if corr is not None:
                results[h_sec].append(corr)
                used_this_window = True

        if used_this_window:
            windows_used += 1

    log.info(f"Used {windows_used} windows with sufficient data")
    if windows_used == 0:
        log.warning("No windows had enough overlapping data. Let both pollers run longer.")
        sys.exit(0)

    print("\nGap -> future Kalshi price-change correlation, by horizon:")
    print("(positive correlation = gap predicts convergence: Kalshi price")
    print(" moves toward closing the Coinbase-implied gap over that horizon)")
    avg_by_horizon = {}
    for h_sec in horizon_seconds_list:
        corrs = results[h_sec]
        if not corrs:
            continue
        avg = float(np.mean(corrs))
        avg_by_horizon[h_sec] = avg
        bar = "#" * int(abs(avg) * 50)
        print(f"  {h_sec:4d}s (n={len(corrs):3d} windows) : {avg:+.4f}  {bar}")

    if avg_by_horizon:
        best_h = max(avg_by_horizon, key=lambda k: avg_by_horizon[k])
        print(f"\nStrongest convergence signal at {best_h}s horizon "
              f"(correlation = {avg_by_horizon[best_h]:+.4f})")
        print(f"\nCaveat: gap and price-change series are both random-walk-like, which can "
              f"produce spurious correlation in the ~0.2-0.3 range even with NO real "
              f"relationship (confirmed via synthetic null testing during development). "
              f"Treat correlations under ~0.35 as inconclusive noise, not evidence of a "
              f"real, tradeable convergence effect -- the bar here is higher than for a "
              f"typical correlation test.")


if __name__ == "__main__":
    main()
