"""
Genuinely different question from mean_surge_outcome_analysis.py: does
window N's mean-position-vs-strike + late-2-minute-move signal predict
window N+1's SEPARATE outcome -- not window N's own outcome (which is
close to circular, since the strike IS the settlement threshold and
settlement is itself a late-window average -- a window's own mean sitting
above its own strike is almost definitionally tied to that same window
resolving up).

This only has a real answer if there's an actual cross-window momentum or
reversion effect. No structural reason it should be true -- that's what
makes it worth checking, unlike the intra-window version.

Same safeguards as the other window-sequence scripts tonight: only counts
a (window N, window N+1) pair when they're genuinely adjacent (exactly 15
minutes apart), and applies the same pre-fix cutoff and Kalshi-authoritative-
outcome preference established earlier.

Usage:
    python3 cross_window_mean_surge_analysis.py [--windows 300] [--late-window-seconds 120]
"""
from __future__ import annotations
import argparse
import logging
from datetime import datetime, timedelta

import boto3
from boto3.dynamodb.conditions import Key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cross_window_mean_surge_analysis")

REGION = "us-east-1"
PRE_FIX_CUTOFF_TS = 1787118936.0  # 2026-08-19T05:55:36+00:00 UTC -- same cutoff as
                                    # mean_surge_outcome_analysis.py (book-logging fix)


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


def get_all_windows(region: str = REGION) -> list[dict]:
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table("btc_windows")
    items = []
    scan_kwargs = {}
    while True:
        resp = table.scan(**scan_kwargs)
        items.extend(resp["Items"])
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    results = []
    for i in items:
        if i.get("source") == "backfill":
            continue
        true_outcome = i.get("kalshi_true_outcome")
        outcome = true_outcome or i.get("outcome")
        strike = i.get("kalshi_true_strike") or i.get("open_price")
        if outcome not in ("up", "down") or strike is None:
            continue
        window_ts = datetime.fromisoformat(i["window_id"]).timestamp()
        if window_ts < PRE_FIX_CUTOFF_TS:
            continue
        results.append({"window_id": i["window_id"], "outcome": outcome, "strike": float(strike)})

    results.sort(key=lambda x: x["window_id"])
    return results


def analyze_window(window_id: str, strike: float, late_window_seconds: float, region: str) -> dict | None:
    """Same computation as mean_surge_outcome_analysis.py's analyze_window --
    THIS window's own mean position and late move, using only THIS
    window's own ticks."""
    ticks = get_window_ticks("btc_ticks", window_id, region=region)
    if len(ticks) < 5:
        return None

    window_start_ts = datetime.fromisoformat(window_id).timestamp()
    window_end_ts = window_start_ts + 900.0
    late_boundary_ts = window_end_ts - late_window_seconds

    early_prices = [float(t["price"]) for t in ticks if float(t["timestamp"]) < late_boundary_ts]
    late_ticks = [t for t in ticks if float(t["timestamp"]) >= late_boundary_ts]

    if len(early_prices) < 3 or len(late_ticks) < 2:
        return None

    mean_price = sum(early_prices) / len(early_prices)
    mean_position = "above" if mean_price > strike else "below"
    late_change = float(late_ticks[-1]["price"]) - early_prices[-1]

    return {"mean_position": mean_position, "late_change": late_change}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=300)
    parser.add_argument("--late-window-seconds", type=float, default=120.0)
    parser.add_argument("--region", type=str, default=REGION)
    args = parser.parse_args()

    log.info("Pulling window outcomes...")
    all_windows = get_all_windows(region=args.region)
    windows = all_windows[-args.windows:] if args.windows else all_windows
    log.info(f"Analyzing {len(windows)} windows (post-cutoff only) for cross-window prediction")

    buckets = {}
    skipped_gaps = 0
    used = 0

    for i in range(len(windows) - 1):
        w_now = windows[i]
        w_next = windows[i + 1]

        # Only use this pair if they're genuinely adjacent -- same safeguard
        # as outcome_streak_analysis.py, critical given this session's own
        # history of gaps from restarts/instability.
        t_now = datetime.fromisoformat(w_now["window_id"])
        t_next = datetime.fromisoformat(w_next["window_id"])
        if (t_next - t_now) != timedelta(minutes=15):
            skipped_gaps += 1
            continue

        signal = analyze_window(w_now["window_id"], w_now["strike"], args.late_window_seconds, args.region)
        if signal is None:
            continue

        late_sign = "positive" if signal["late_change"] > 0 else ("negative" if signal["late_change"] < 0 else "flat")
        key = (signal["mean_position"], late_sign)
        buckets.setdefault(key, []).append(w_next["outcome"])
        used += 1

    log.info(f"{used} valid adjacent (window N, window N+1) pairs used, {skipped_gaps} skipped due to a gap\n")

    print(f"=== Window N's mean/late signal vs window N+1's SEPARATE outcome (n={used}) ===")
    print(f"(if this shows ~50% across all buckets, there's no real cross-window effect --")
    print(f" that would be the honest, expected null result, not a failure of the test)\n")
    for key, outcomes in sorted(buckets.items()):
        n = len(outcomes)
        n_up = sum(1 for o in outcomes if o == "up")
        pct_up = 100 * n_up / n if n else 0
        print(f"  window N: mean {key[0]} strike, late change {key[1]}: n={n:4d}  "
              f"% window N+1 resolved UP={pct_up:5.1f}%")


if __name__ == "__main__":
    main()
