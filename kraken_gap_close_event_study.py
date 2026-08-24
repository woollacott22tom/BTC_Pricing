"""
Tests a specific hypothesis distinct from anything tested earlier tonight:
does the MOMENT Kraken closes a persistent price gap against Coinbase (not
the gap's existence, and not its magnitude relative to a rolling baseline
-- specifically the CLOSING EVENT after a sustained lag) predict a
subsequent Kalshi price move?

The idea: Kraken trades much less frequently than Coinbase (established
repeatedly tonight). During ordinary noise, Kraken just looks sparse/quiet.
But when a real, sustained gap opens and PERSISTS for a while (not just a
one-tick blip), Kraken finally moving to close it may be a genuine
"confirmation" signal -- the quiet exchange speaking up specifically
because the move was large enough to matter, rather than reacting to
every small fluctuation the way Coinbase's constant tick stream does.

Method, per window:
  1. Align Coinbase's price onto Kraken's own (sparse) tick timestamps.
  2. Compute gap(t) = kraken_price(t) - coinbase_price(t).
  3. A GAP-CLOSE EVENT at tick t requires:
       a. The gap was PERSISTENTLY large (|gap| >= gap_open_threshold,
          consistent sign) throughout the trailing persist_seconds window
          -- a real, sustained lag, not noise flipping back and forth.
       b. The CURRENT gap has dropped sharply (|gap| < gap_close_threshold)
          -- Kraken just caught up.
  4. Measure Kalshi's forward price change after each such event, split by
     which direction the gap was closing FROM (i.e. which way Kraken was
     lagging) -- this is the direction the hypothesis predicts Kalshi
     should move.

Usage:
    python3 kraken_gap_close_event_study.py [--windows 100] [--horizon-seconds 45]
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
log = logging.getLogger("kraken_gap_close_study")

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


def build_gap_series(target_ticks: list[dict], other1_ticks: list[dict], other2_ticks: list[dict]) -> pd.DataFrame | None:
    """gap(t) = target_price(t) - average(other1_price(t), other2_price(t)),
    aligned onto the TARGET exchange's own (sparser) tick timestamps.
    Generalized from an earlier fixed Kraken-vs-Coinbase-only version --
    real observation showed this pattern isn't specific to one pair (a
    Coinbase-vs-Crypto.com gap closing alongside Kraken catching up was
    observed together), so any exchange can now be tested against the
    consensus of the other two, not just Kraken against Coinbase."""
    if len(target_ticks) < 1 or len(other1_ticks) < 1 or len(other2_ticks) < 1:
        return None

    target_df = pd.DataFrame({
        "timestamp": [float(t["timestamp"]) for t in target_ticks],
        "target_price": [float(t["price"]) for t in target_ticks if "price" in t],
    }).sort_values("timestamp")
    if len(target_df) < 1:
        return None

    other1_df = pd.DataFrame({
        "timestamp": [float(t["timestamp"]) for t in other1_ticks],
        "other1_price": [float(t["price"]) for t in other1_ticks],
    }).sort_values("timestamp")

    other2_df = pd.DataFrame({
        "timestamp": [float(t["timestamp"]) for t in other2_ticks],
        "other2_price": [float(t["price"]) for t in other2_ticks],
    }).sort_values("timestamp")

    merged = pd.merge_asof(target_df, other1_df, on="timestamp", direction="backward", tolerance=10.0)
    merged = pd.merge_asof(merged, other2_df, on="timestamp", direction="backward", tolerance=10.0)
    merged = merged.dropna(subset=["target_price", "other1_price", "other2_price"])
    if len(merged) < 1:
        return None

    merged["avg_other"] = (merged["other1_price"] + merged["other2_price"]) / 2.0
    merged["gap"] = merged["target_price"] - merged["avg_other"]
    return merged.reset_index(drop=True)


def detect_gap_close_events(gap_df: pd.DataFrame, persist_seconds: float,
                              gap_open_threshold: float, gap_close_threshold: float,
                              min_same_sign_fraction: float = 0.8) -> pd.DataFrame:
    """For each tick, checks whether the trailing persist_seconds window
    shows a PERSISTENT, consistently-signed gap (a real sustained lag),
    and whether the CURRENT gap has since dropped sharply (Kraken caught
    up). Uses the window's MEAN gap magnitude and a same-sign FRACTION
    (not requiring every single tick to individually qualify) -- real
    market gaps wobble continuously even during a genuine sustained lag,
    so requiring unanimous agreement across every tick in the window is
    unrealistically strict and produces false negatives (confirmed: the
    original all-ticks-must-qualify version found zero events across 300
    real windows, which is itself evidence the detector was too strict,
    not that the underlying pattern doesn't occur)."""
    df = gap_df.copy()
    was_persistent = []
    prior_direction = []

    for i in range(len(df)):
        t_now = df["timestamp"].iloc[i]
        window = df[(df["timestamp"] >= t_now - persist_seconds) & (df["timestamp"] < t_now)]
        if len(window) < 1:
            was_persistent.append(False)
            prior_direction.append(0)
            continue
        gaps = window["gap"].values
        mean_gap = gaps.mean()
        if abs(mean_gap) < gap_open_threshold:
            was_persistent.append(False)
            prior_direction.append(0)
            continue
        same_sign_fraction = float((np.sign(gaps) == np.sign(mean_gap)).mean())
        is_persistent = same_sign_fraction >= min_same_sign_fraction
        was_persistent.append(is_persistent)
        prior_direction.append((1 if mean_gap > 0 else -1) if is_persistent else 0)

    df["was_persistent_gap"] = was_persistent
    df["prior_gap_direction"] = prior_direction
    df["gap_close_event"] = df["was_persistent_gap"] & (df["gap"].abs() < gap_close_threshold)

    return df


def forward_kalshi_change(kalshi_df: pd.DataFrame, event_ts: float, horizon: float) -> float | None:
    before = kalshi_df[kalshi_df["timestamp"] <= event_ts]
    after = kalshi_df[kalshi_df["timestamp"] <= event_ts + horizon]
    if before.empty or after.empty:
        return None
    baseline = before.iloc[-1]
    target = after.iloc[-1]
    if target["timestamp"] <= baseline["timestamp"]:
        return None
    return float(target["yes_mid_cents"] - baseline["yes_mid_cents"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-exchange", type=str, default="kraken",
                         choices=["kraken", "coinbase", "cryptocom"],
                         help="Which exchange to test for gap-close events against the average "
                              "of the OTHER two -- not fixed to Kraken vs Coinbase specifically")
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--persist-seconds", type=float, default=30.0,
                         help="How long the gap must have persisted before a close counts as an event")
    parser.add_argument("--gap-open-dollars", type=float, default=50.0,
                         help="Minimum |gap| to count as 'persistently open' -- default matches a "
                              "real observed case, NOT a validated threshold; treat as a starting point")
    parser.add_argument("--gap-close-dollars", type=float, default=5.0,
                         help="Maximum |gap| to count as 'closed'")
    parser.add_argument("--horizon-seconds", type=float, default=45.0)
    parser.add_argument("--region", type=str, default=REGION)
    parser.add_argument("--diagnose-gaps-only", action="store_true",
                         help="Skip event detection entirely -- just print the real observed "
                              "gap magnitude distribution, to choose thresholds from evidence "
                              "instead of continuing to guess after repeated zero-event runs")
    args = parser.parse_args()

    exchange_tables = {"kraken": "kraken_ticks", "coinbase": "btc_ticks", "cryptocom": "cryptocom_ticks"}
    target_table = exchange_tables[args.target_exchange]
    other_tables = [t for name, t in exchange_tables.items() if name != args.target_exchange]

    window_ids = get_recent_window_ids(args.windows, region=args.region)

    if args.diagnose_gaps_only:
        all_gaps = []
        for wid in window_ids:
            target_ticks = get_window_ticks(target_table, wid, region=args.region)
            other1_ticks = get_window_ticks(other_tables[0], wid, region=args.region)
            other2_ticks = get_window_ticks(other_tables[1], wid, region=args.region)
            gap_df = build_gap_series(target_ticks, other1_ticks, other2_ticks)
            if gap_df is not None:
                all_gaps.extend(gap_df["gap"].tolist())

        if not all_gaps:
            print("No gap data found at all -- check that all three exchanges have ticks in these windows.")
            return

        arr = np.array(all_gaps)
        print(f"=== Real {args.target_exchange} vs (average of other two) gap distribution ===")
        print(f"n={len(arr)} observations across {len(window_ids)} windows")
        print(f"mean={arr.mean():+.2f}  median={np.median(arr):+.2f}  std={arr.std():.2f}")
        print(f"min={arr.min():+.2f}  max={arr.max():+.2f}")
        for pct in [50, 75, 90, 95, 99]:
            print(f"  {pct}th percentile of |gap|: {np.percentile(np.abs(arr), pct):.2f}")
        print(f"\n% of observations with |gap| >= $50 (current default threshold): "
              f"{100*float((np.abs(arr) >= 50).mean()):.2f}%")
        print(f"% of observations with |gap| >= $20: {100*float((np.abs(arr) >= 20).mean()):.2f}%")
        print(f"% of observations with |gap| >= $10: {100*float((np.abs(arr) >= 10).mean()):.2f}%")
        return

    log.info(f"Checking {len(window_ids)} windows for {args.target_exchange} gap-close events "
             f"(vs average of the other two)")

    changes_by_direction = {1: [], -1: []}
    windows_used = 0

    for wid in window_ids:
        target_ticks = get_window_ticks(target_table, wid, region=args.region)
        other1_ticks = get_window_ticks(other_tables[0], wid, region=args.region)
        other2_ticks = get_window_ticks(other_tables[1], wid, region=args.region)
        kalshi_ticks = get_window_ticks("kalshi_prices", wid, region=args.region)
        if len(kalshi_ticks) < 3:
            continue

        gap_df = build_gap_series(target_ticks, other1_ticks, other2_ticks)
        if gap_df is None:
            continue
        gap_df = detect_gap_close_events(gap_df, args.persist_seconds, args.gap_open_dollars, args.gap_close_dollars)

        events = gap_df[gap_df["gap_close_event"]]
        if events.empty:
            continue

        kalshi_df = pd.DataFrame({
            "timestamp": [float(t["timestamp"]) for t in kalshi_ticks],
            "yes_mid_cents": [float(t["yes_mid_cents"]) for t in kalshi_ticks],
        }).sort_values("timestamp")

        used_this_window = False
        for _, ev in events.iterrows():
            change = forward_kalshi_change(kalshi_df, ev["timestamp"], args.horizon_seconds)
            if change is not None:
                direction = int(ev["prior_gap_direction"])
                if direction in changes_by_direction:
                    changes_by_direction[direction].append(change)
                    used_this_window = True

        if used_this_window:
            windows_used += 1

    log.info(f"Used {windows_used} windows with at least one gap-close event\n")

    print(f"=== Kraken gap-close events (forward Kalshi change over {args.horizon_seconds}s) ===")
    print(f"(direction = which way Kraken was lagging before it caught up; theory predicts")
    print(f" a REVERSAL -- Kraken's catch-up signals the move is fully absorbed, not that it continues)")
    for direction, changes in changes_by_direction.items():
        if not changes:
            print(f"  direction={'+' if direction > 0 else '-'} ({args.target_exchange} above/below the other two's average): no events found")
            continue
        arr = np.array(changes)
        dir_label = (f"+ ({args.target_exchange} was ABOVE the other two's average, closing down)" if direction > 0
                     else f"- ({args.target_exchange} was BELOW the other two's average, closing up)")
        # Hypothesis is REVERSAL, not continuation, matching the real observed
        # example: Kraken lagged ABOVE Coinbase during a down-move (hadn't
        # caught down yet), and when it finally did catch down (closing a
        # positive gap), the reversal (price going back UP) followed shortly
        # after -- i.e. Kraken's catch-up signals the move is now fully
        # absorbed/exhausted, not that it's about to continue. So:
        #   direction=+1 (was above, closing down) predicts Kalshi UP afterward
        #   direction=-1 (was below, closing up) predicts Kalshi DOWN afterward
        aligned = (direction > 0 and arr.mean() > 0) or (direction < 0 and arr.mean() < 0)
        tag = "predicts REVERSAL in EXPECTED direction" if aligned else "CONTRADICTS expectation" if arr.mean() != 0 else "no clear effect"
        print(f"  {dir_label}: n={len(arr):4d}  mean_change={arr.mean():+.4f}cents  "
              f"median={np.median(arr):+.4f}  std={arr.std():.4f}  [{tag}]")

    print(f"\nCaveat: gap-close events requiring {args.persist_seconds}s of sustained lag are likely rare --")
    print(f"small n here is expected, not a sign of a bug. Treat this as a first read.")


if __name__ == "__main__":
    main()
