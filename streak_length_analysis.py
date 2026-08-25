"""
outcome_streak_analysis.py showed P(up|prev=up) and P(down|prev=down) both
sit close to 50% -- but that only tests streak length 1 (does the outcome
right after a single up/down repeat). It's possible the simple one-step
transition is noise while LONGER streaks behave differently -- e.g. streaks
becoming MORE likely to continue as they build (momentum), or increasingly
likely to snap the longer they run (mean reversion). Both are real,
well-documented patterns in various markets; neither is visible in a
single lag-1 test.

This walks the same chronological outcome sequence and, at each point,
tracks how long the CURRENT streak already is before checking whether the
next window continues or flips it -- then buckets the continuation rate BY
streak length, so length-dependent effects (if any) become visible.

Same safeguards as the other window-sequence scripts: only walks across
genuinely adjacent (15-minute) windows, using Kalshi's authoritative
outcome where available. A gap in the data resets streak tracking
entirely (can't trust continuity across an unknown stretch) rather than
just skipping the one affected pair.

Usage:
    python3 streak_length_analysis.py [--limit 500]
"""
from __future__ import annotations
import argparse
import logging
from datetime import datetime, timedelta

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("streak_length_analysis")

REGION = "us-east-1"
WINDOWS_TABLE = "btc_windows"


def get_all_windows_with_outcomes(region: str = REGION) -> list[dict]:
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(WINDOWS_TABLE)
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
        if outcome not in ("up", "down"):
            continue
        results.append({"window_id": i["window_id"], "outcome": outcome})

    results.sort(key=lambda x: x["window_id"])
    return results


def compute_streak_continuation_by_length(outcomes: list[dict]) -> dict[int, list[bool]]:
    """For each window (after the first), determines the length of the
    streak ending at the PREVIOUS window, then records whether THIS window
    continued (True) or flipped (False) that streak. Buckets results by
    streak length. Gaps in the data (non-adjacent windows) reset streak
    tracking entirely -- a streak can't be trusted to span an unknown gap."""
    results: dict[int, list[bool]] = {}
    streak_len = None
    streak_outcome = None

    for i in range(1, len(outcomes)):
        t_prev = datetime.fromisoformat(outcomes[i - 1]["window_id"])
        t_cur = datetime.fromisoformat(outcomes[i]["window_id"])
        if (t_cur - t_prev) != timedelta(minutes=15):
            streak_len = None
            streak_outcome = None
            continue

        if streak_len is None:
            streak_len = 1
            streak_outcome = outcomes[i - 1]["outcome"]

        continued = (outcomes[i]["outcome"] == streak_outcome)
        results.setdefault(streak_len, []).append(continued)

        if continued:
            streak_len += 1
        else:
            streak_len = 1
            streak_outcome = outcomes[i]["outcome"]

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--region", type=str, default=REGION)
    args = parser.parse_args()

    log.info("Pulling all window outcomes...")
    outcomes = get_all_windows_with_outcomes(region=args.region)
    if args.limit:
        outcomes = outcomes[-args.limit:]
    log.info(f"Analyzing {len(outcomes)} windows")

    results = compute_streak_continuation_by_length(outcomes)

    print(f"=== Streak continuation probability, by current streak length ===")
    print(f"(streak_len=1 means 'just one up/down so far' -- matches outcome_streak_analysis.py's")
    print(f" ~52%/48% result. Longer lengths test whether that changes as streaks build.)\n")
    for length in sorted(results.keys()):
        vals = results[length]
        n = len(vals)
        pct_continued = 100 * sum(vals) / n if n else 0
        bar = "#" * int(pct_continued / 2)
        print(f"  streak_len={length:2d}  n={n:4d}  % continued={pct_continued:5.1f}%  {bar}")

    print(f"\nCaveat: n shrinks fast at longer streak lengths (long streaks are rare by")
    print(f"definition) -- treat single-digit-n rows as noise, not signal.")


if __name__ == "__main__":
    main()
