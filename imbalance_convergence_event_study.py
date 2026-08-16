"""
Tests a specific, discrete-event hypothesis rather than a continuous
correlation: does a MOMENT where an exchange's order book tiers resolve
from disagreement (e.g. top-10 positive, deeper tiers negative) into full
consensus (all tiers the same sign) predict Kalshi's subsequent price
movement in that direction? Also checks simple sign-flips on the top-10
book_imbalance alone.

A plain correlation on the imbalance LEVEL (lead_lag_analysis.py's
--signal book_imbalance mode) would wash this out -- most of the time
nothing is transitioning, and that steady-state noise dilutes the signal
from the rare, meaningful convergence moments. This is an event-study
instead: find the specific transition ticks, then average Kalshi's
forward price change conditional on that event's direction.

Method, per window with both exchange + Kalshi data:
  1. For each exchange tick, compute the sign of each depth tier
     (imbalance_5, book_imbalance [top-10], imbalance_20, imbalance_50).
  2. A tick is in "consensus" if all four tiers share the same sign.
  3. A CONVERGENCE EVENT is a tick where the previous tick was NOT in
     consensus but this tick IS -- i.e. the book just resolved internal
     disagreement into agreement. The event's direction is whichever sign
     all tiers now share.
  4. A FLIP EVENT is simpler: just the top-10 book_imbalance crossing zero
     between consecutive ticks.
  5. For each event, measure Kalshi's price change from just before the
     event to `horizon` seconds after it (via as-of lookups against
     Kalshi's own tick series). Aggregate by event direction across all
     windows: does convergence-to-negative predict Kalshi falling
     afterward, and convergence-to-positive predict Kalshi rising?

Usage:
    python3 imbalance_convergence_event_study.py --exchange cryptocom [--horizon-seconds 15] [--windows 100]
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
log = logging.getLogger("convergence_event_study")

REGION = os.environ.get("AWS_REGION", "us-east-1")
EXCHANGE_TABLES = {"coinbase": "btc_ticks", "kraken": "kraken_ticks", "cryptocom": "cryptocom_ticks"}
KALSHI_TABLE = "kalshi_prices"

TIER_FIELDS = ["feat_imbalance_5", "feat_book_imbalance", "feat_imbalance_20", "feat_imbalance_50"]
_debug_zero_count = 0


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


def build_tier_state_df(exchange_ticks: list[dict]) -> pd.DataFrame | None:
    rows = []
    for t in exchange_ticks:
        if not all(f in t and t[f] is not None for f in TIER_FIELDS):
            continue
        vals = [float(t[f]) for f in TIER_FIELDS]
        rows.append({"timestamp": float(t["timestamp"]), **{f: v for f, v in zip(TIER_FIELDS, vals)}})
    if len(rows) < 3:
        return None

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    signs = np.sign(df[TIER_FIELDS].values)
    all_positive = (signs > 0).all(axis=1)
    all_negative = (signs < 0).all(axis=1)
    df["is_consensus"] = all_positive | all_negative
    df["consensus_direction"] = np.where(all_positive, 1, np.where(all_negative, -1, 0))
    df["top10_sign"] = np.sign(df["feat_book_imbalance"])

    df["prev_is_consensus"] = df["is_consensus"].shift(1)
    df["prev_top10_sign"] = df["top10_sign"].shift(1)

    df["convergence_event"] = (~df["prev_is_consensus"].fillna(True)) & df["is_consensus"]
    df["flip_event"] = df["prev_top10_sign"].notna() & (df["top10_sign"] != 0) & \
                        (df["prev_top10_sign"] != 0) & (df["top10_sign"] != df["prev_top10_sign"])

    return df


def add_surge_events(df: pd.DataFrame, low_threshold: float, high_threshold: float, delta_threshold: float) -> pd.DataFrame:
    """Two kinds of 'significant swing', OR'd together:
      1. Weak-to-strong: |value| goes from below low_threshold to above
         high_threshold (the original definition).
      2. Delta-based: the magnitude jumps by more than delta_threshold
         between consecutive ticks, REGARDLESS of the starting level --
         this catches swings on an already-elevated base (e.g. 0.4 -> 0.8),
         which the weak-to-strong-only definition would miss entirely."""
    df = df.copy()
    for field in TIER_FIELDS:
        prev = df[field].shift(1)
        curr = df[field]
        was_weak = prev.notna() & (prev.abs() < low_threshold)
        now_strong = curr.abs() > high_threshold
        weak_to_strong = was_weak & now_strong

        magnitude_jump = prev.notna() & ((curr.abs() - prev.abs()) > delta_threshold)

        df[f"surge_event_{field}"] = weak_to_strong | magnitude_jump
        df[f"surge_direction_{field}"] = np.sign(curr)
    return df


def compute_kalshi_trend_state(kalshi_df: pd.DataFrame, event_ts: float, lookback: float, trend_threshold: float) -> str:
    """Classifies whether Kalshi's OWN price was already trending up, down,
    or flat in the `lookback` seconds immediately before the event -- lets
    us test continuation (event aligns with an existing trend) vs reversal
    (event fights an existing trend) separately, rather than only looking
    at raw forward change in isolation."""
    window = kalshi_df[(kalshi_df["timestamp"] >= event_ts - lookback) & (kalshi_df["timestamp"] <= event_ts)]
    if len(window) < 2:
        return "unknown"
    change = window["yes_mid_cents"].iloc[-1] - window["yes_mid_cents"].iloc[0]
    if change > trend_threshold:
        return "kalshi_uptrend"
    elif change < -trend_threshold:
        return "kalshi_downtrend"
    return "kalshi_flat"


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
    parser.add_argument("--exchange", type=str, default="cryptocom", choices=list(EXCHANGE_TABLES.keys()))
    parser.add_argument("--horizon-seconds", type=float, default=15.0)
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--surge-low", type=float, default=0.15,
                         help="A tier is considered 'weak' below this |imbalance| before a surge")
    parser.add_argument("--surge-high", type=float, default=0.5,
                         help="A tier is considered 'strong' above this |imbalance| after a surge")
    parser.add_argument("--surge-delta", type=float, default=0.3,
                         help="Also counts as a surge if |imbalance| jumps by more than this, "
                              "even starting from an already-elevated level")
    parser.add_argument("--trend-lookback-seconds", type=float, default=15.0,
                         help="How far back to look when classifying whether Kalshi was already trending")
    parser.add_argument("--trend-threshold", type=float, default=0.5,
                         help="Minimum cents of prior Kalshi movement to count as an existing trend")
    parser.add_argument("--region", type=str, default=REGION)
    args = parser.parse_args()

    exchange_table = EXCHANGE_TABLES[args.exchange]
    window_ids = get_recent_window_ids(args.windows, region=args.region)
    log.info(f"Checking {len(window_ids)} windows for {args.exchange} tier-convergence events")

    convergence_changes = {1: [], -1: []}
    flip_changes = {1: [], -1: []}
    surge_changes = {field: {1: [], -1: []} for field in TIER_FIELDS}
    # Cross-tabulated by (surge direction, Kalshi's pre-existing trend state) --
    # this is what actually tests continuation-vs-reversal, not just raw forward change
    surge_by_trend = {field: {} for field in TIER_FIELDS}
    windows_used = 0

    for wid in window_ids:
        exchange_ticks = get_window_ticks(exchange_table, wid, region=args.region)
        kalshi_ticks = get_window_ticks(KALSHI_TABLE, wid, region=args.region)
        if len(kalshi_ticks) < 3:
            continue

        state_df = build_tier_state_df(exchange_ticks)
        if state_df is None:
            continue
        state_df = add_surge_events(state_df, args.surge_low, args.surge_high, args.surge_delta)

        kalshi_df = pd.DataFrame({
            "timestamp": [float(t["timestamp"]) for t in kalshi_ticks],
            "yes_mid_cents": [float(t["yes_mid_cents"]) for t in kalshi_ticks],
        }).sort_values("timestamp")

        used_this_window = False

        conv_events = state_df[state_df["convergence_event"]]
        for _, ev in conv_events.iterrows():
            change = forward_kalshi_change(kalshi_df, ev["timestamp"], args.horizon_seconds)
            if change is not None:
                convergence_changes[int(ev["consensus_direction"])].append(change)
                used_this_window = True

        flip_events = state_df[state_df["flip_event"]]
        for _, ev in flip_events.iterrows():
            change = forward_kalshi_change(kalshi_df, ev["timestamp"], args.horizon_seconds)
            if change is not None:
                flip_changes[int(ev["top10_sign"])].append(change)
                used_this_window = True

        for field in TIER_FIELDS:
            surge_events = state_df[state_df[f"surge_event_{field}"]]
            for _, ev in surge_events.iterrows():
                change = forward_kalshi_change(kalshi_df, ev["timestamp"], args.horizon_seconds)
                if change is None:
                    continue
                direction = int(ev[f"surge_direction_{field}"])
                if direction not in surge_changes[field]:
                    continue
                surge_changes[field][direction].append(change)

                trend = compute_kalshi_trend_state(
                    kalshi_df, ev["timestamp"], args.trend_lookback_seconds, args.trend_threshold
                )
                key = (direction, trend)
                surge_by_trend[field].setdefault(key, []).append(change)
                used_this_window = True

                # DEEP DEBUG: for the first few exact-zero changes on a
                # trending (non-flat) event, print the actual underlying
                # rows -- forward_kalshi_change doesn't look at trend at
                # all, so if zeros cluster specifically on trend!=flat,
                # something about the interaction needs to be seen directly.
                global _debug_zero_count
                if change == 0.0 and trend in ("kalshi_uptrend", "kalshi_downtrend") and _debug_zero_count < 5:
                    ev_ts = ev["timestamp"]
                    before = kalshi_df[kalshi_df["timestamp"] <= ev_ts]
                    after = kalshi_df[kalshi_df["timestamp"] <= ev_ts + args.horizon_seconds]
                    trend_window = kalshi_df[(kalshi_df["timestamp"] >= ev_ts - args.trend_lookback_seconds) &
                                              (kalshi_df["timestamp"] <= ev_ts)]
                    print(f"    [DEEP DEBUG #{_debug_zero_count}] window={wid} event_ts={ev_ts:.2f} trend={trend}")
                    print(f"      trend_window rows (ts, yes_mid_cents): "
                          f"{list(zip(trend_window['timestamp'].round(2), trend_window['yes_mid_cents']))}")
                    print(f"      baseline row: ts={before.iloc[-1]['timestamp']:.2f} "
                          f"price={before.iloc[-1]['yes_mid_cents']}")
                    print(f"      target row:   ts={after.iloc[-1]['timestamp']:.2f} "
                          f"price={after.iloc[-1]['yes_mid_cents']}")
                    _debug_zero_count += 1

        if used_this_window:
            windows_used += 1

    log.info(f"Used {windows_used} windows with at least one usable event\n")

    def report(label: str, changes_by_dir: dict[int, list[float]]):
        print(f"=== {label} (forward Kalshi change over {args.horizon_seconds}s) ===")
        for direction, changes in changes_by_dir.items():
            if not changes:
                print(f"  direction={'+' if direction > 0 else '-'}: no events found")
                continue
            arr = np.array(changes)
            if (direction > 0 and arr.mean() > 0) or (direction < 0 and arr.mean() < 0):
                expected = "predicts move in EXPECTED direction"
            elif arr.mean() == 0:
                expected = "no clear effect"
            else:
                expected = "CONTRADICTS expectation"
            print(f"  direction={'+' if direction > 0 else '-'}: n={len(arr):4d}  "
                  f"mean_change={arr.mean():+.4f}cents  median={np.median(arr):+.4f}  "
                  f"std={arr.std():.4f}  [{expected}]")
        print()

    report("Tier convergence events (disagreement -> full consensus)", convergence_changes)
    report("Top-10 sign-flip events", flip_changes)

    tier_labels = {
        "feat_imbalance_5": "top-5", "feat_book_imbalance": "top-10",
        "feat_imbalance_20": "top-20", "feat_imbalance_50": "top-50",
    }
    print(f"Surge events (weak |imb|<{args.surge_low} -> strong |imb|>{args.surge_high}, "
          f"OR any jump >{args.surge_delta} regardless of starting level), by tier:")
    print("(compares which depth's surges are most predictive)")
    for field in TIER_FIELDS:
        report(f"  {tier_labels[field]} surge events", surge_changes[field])

    print(f"Surge events split by Kalshi's PRE-EXISTING trend (trailing {args.trend_lookback_seconds}s "
          f"before the event, threshold {args.trend_threshold}cents):")
    print("(tests continuation -- surge aligns with an existing trend -- vs reversal, or no trend at all)")
    for field in TIER_FIELDS:
        print(f"  --- {tier_labels[field]} ---")
        for (direction, trend), changes in sorted(surge_by_trend[field].items()):
            if not changes:
                continue
            arr = np.array(changes)
            dir_label = "+" if direction > 0 else "-"
            aligned = (direction > 0 and trend == "kalshi_uptrend") or (direction < 0 and trend == "kalshi_downtrend")
            tag = "CONTINUATION setup" if aligned else ("REVERSAL setup" if trend != "kalshi_flat" and trend != "unknown" else "no prior trend")
            print(f"    dir={dir_label}  trend={trend:16s}  n={len(arr):4d}  "
                  f"mean_change={arr.mean():+.4f}cents  [{tag}]")
            if trend in ("kalshi_uptrend", "kalshi_downtrend") and abs(arr.mean()) < 1e-9:
                sample = changes[:8]
                print(f"      DIAGNOSTIC (mean suspiciously exactly zero) -- raw sample: {sample}")
    print()

    print("Caveat: small n per direction means these means can be noisy -- treat this as a first")
    print("read, not a final verdict. If one direction shows a consistent, sizable mean_change")
    print("while the other doesn't, or both directions point the SAME way (which would be a red")
    print("flag, e.g. a labeling bug), that's more informative than either number alone.")


if __name__ == "__main__":
    main()
