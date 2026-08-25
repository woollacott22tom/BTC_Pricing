"""
For each window: was the AVERAGE price (over the first 13 minutes, i.e.
excluding the final 2 minutes) above or below the strike? And what was
the price CHANGE over the final 2 minutes specifically? Then checks how
those two relate to the window's actual outcome -- directly testing
whether a late surge can override what the bulk of the window suggested
(e.g. "mean below strike, but a late surge pushes it to resolve up").

Also reports how each bucket relates to CONTINUATION vs CHANGE from the
previous window's outcome, connecting back to outcome_streak_analysis.py's
52/48 result -- to see whether mean-position + late-change explains any of
that noise, or whether it's independent of it.

Uses Coinbase's own tick series (btc_ticks) as the price reference, and
the SAME pre-fix cutoff established in train.py to exclude data collected
before the book-logging fix (see ORDER_BOOK_FIX_CUTOFF_TS comment) --
average-price and late-window-price-change calculations both depend on
having a complete, correctly-sampled tick history throughout the window,
which pre-cutoff data does not reliably have.

Usage:
    python3 mean_surge_outcome_analysis.py [--windows 300] [--late-window-seconds 120]
"""
from __future__ import annotations
import argparse
import logging
from datetime import datetime, timedelta

import boto3
from boto3.dynamodb.conditions import Key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mean_surge_outcome_analysis")

REGION = "us-east-1"

# Same cutoff as train.py's ORDER_BOOK_FIX_CUTOFF_TS -- the confirmed
# deploy time of the Coinbase + Kraken book-logging fix. Before this,
# ticks were only logged when a trade fired, so a window's average price
# and its final-2-minute price change would both be computed from
# incomplete, coarsely-sampled data -- not a fair test of either signal.
PRE_FIX_CUTOFF_TS = 1787118936.0  # 2026-08-19T05:55:36+00:00 UTC


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
        strike = i.get("kalshi_true_strike") or i.get("strike_price")
        if outcome not in ("up", "down") or strike is None:
            continue
        window_ts = datetime.fromisoformat(i["window_id"]).timestamp()
        if window_ts < PRE_FIX_CUTOFF_TS:
            continue
        results.append({
            "window_id": i["window_id"],
            "outcome": outcome,
            "strike": float(strike),
        })

    results.sort(key=lambda x: x["window_id"])
    return results


def analyze_window(window_id: str, strike: float, late_window_seconds: float, region: str) -> dict | None:
    ticks = get_window_ticks("btc_ticks", window_id, region=region)
    if len(ticks) < 5:
        return None

    window_start_ts = datetime.fromisoformat(window_id).timestamp()
    window_end_ts = window_start_ts + 900.0  # 15 minutes
    late_boundary_ts = window_end_ts - late_window_seconds

    early_prices = [float(t["price"]) for t in ticks if float(t["timestamp"]) < late_boundary_ts]
    late_ticks = [t for t in ticks if float(t["timestamp"]) >= late_boundary_ts]

    if len(early_prices) < 3 or len(late_ticks) < 2:
        return None

    mean_price = sum(early_prices) / len(early_prices)
    mean_position = "above" if mean_price > strike else "below"

    # Late-window change measured from the LAST early price (right before
    # the late window starts) to the final tick -- this isolates the late
    # move itself, not blended into what's already captured by mean_price.
    late_change = float(late_ticks[-1]["price"]) - early_prices[-1]

    return {"window_id": window_id, "mean_position": mean_position, "late_change": late_change}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=300)
    parser.add_argument("--late-window-seconds", type=float, default=120.0)
    parser.add_argument("--region", type=str, default=REGION)
    args = parser.parse_args()

    log.info("Pulling window outcomes...")
    all_windows = get_all_windows(region=args.region)
    windows = all_windows[-args.windows:] if args.windows else all_windows
    log.info(f"Analyzing {len(windows)} windows (post-cutoff only)")

    analyzed = []
    for i, w in enumerate(windows):
        result = analyze_window(w["window_id"], w["strike"], args.late_window_seconds, args.region)
        if result is None:
            continue
        result["outcome"] = w["outcome"]
        prev_outcome = windows[i - 1]["outcome"] if i > 0 else None
        # only mark continuation/change if the previous window is ACTUALLY adjacent (15 min prior)
        if prev_outcome is not None:
            t_prev = datetime.fromisoformat(windows[i - 1]["window_id"])
            t_cur = datetime.fromisoformat(w["window_id"])
            if (t_cur - t_prev) == timedelta(minutes=15):
                result["continuation"] = (w["outcome"] == prev_outcome)
            else:
                result["continuation"] = None
        else:
            result["continuation"] = None
        analyzed.append(result)

    log.info(f"{len(analyzed)} windows had enough tick data for analysis\n")

    buckets = {}
    for r in analyzed:
        sign = "positive" if r["late_change"] > 0 else ("negative" if r["late_change"] < 0 else "flat")
        key = (r["mean_position"], sign)
        buckets.setdefault(key, []).append(r)

    print(f"=== Mean position (excl. final {args.late_window_seconds:.0f}s) vs late-window price change, vs actual outcome ===")
    print(f"(n={len(analyzed)} windows total)\n")
    for (mean_pos, late_sign), rows in sorted(buckets.items()):
        n = len(rows)
        n_up = sum(1 for r in rows if r["outcome"] == "up")
        pct_up = 100 * n_up / n if n else 0
        override_note = ""
        if mean_pos == "below" and late_sign == "positive":
            override_note = "  <- 'mean below strike, overcome by a late surge' case"
        elif mean_pos == "above" and late_sign == "negative":
            override_note = "  <- 'mean above strike, undercut by a late drop' case"
        print(f"  mean {mean_pos} strike, late change {late_sign}: n={n:4d}  "
              f"% resolved UP={pct_up:5.1f}%{override_note}")

    print(f"\n=== Same buckets, but against CONTINUATION from the previous window ===")
    cont_buckets = {}
    for r in analyzed:
        if r["continuation"] is None:
            continue
        sign = "positive" if r["late_change"] > 0 else ("negative" if r["late_change"] < 0 else "flat")
        key = (r["mean_position"], sign)
        cont_buckets.setdefault(key, []).append(r["continuation"])
    for key, vals in sorted(cont_buckets.items()):
        n = len(vals)
        pct_continued = 100 * sum(vals) / n if n else 0
        print(f"  mean {key[0]} strike, late change {key[1]}: n={n:4d}  "
              f"% continuation (matched previous outcome)={pct_continued:5.1f}%")


if __name__ == "__main__":
    main()
