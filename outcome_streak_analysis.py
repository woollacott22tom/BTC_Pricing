"""
Simple momentum/mean-reversion diagnostic on window OUTCOMES themselves
(not ticks): given the previous 15-min window settled up, how often does
the NEXT window also settle up? And down-after-down?

Prefers Kalshi's own authoritative settlement outcome (written by
kalshi_settlement_reconciliation.py) over our own approximated one,
falling back only when a window hasn't been reconciled yet -- same
precedence already used in train.py.

Correctness note: only counts a transition between two windows that are
ACTUALLY adjacent (exactly 15 minutes apart). Given tonight's own history
of service restarts, there may be real gaps in the data where a window's
outcome is simply missing -- treating two non-adjacent windows as a
"transition" would silently measure something else (a much longer gap),
not a true consecutive-window repeat rate.

Usage:
    python3 outcome_streak_analysis.py [--limit 500]
"""
from __future__ import annotations
import argparse
import logging
from datetime import datetime, timedelta

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("outcome_streak_analysis")

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
        results.append({
            "window_id": i["window_id"],
            "outcome": outcome,
            "authoritative": true_outcome is not None,
        })

    results.sort(key=lambda x: x["window_id"])
    return results


def compute_transition_probabilities(outcomes: list[dict]) -> tuple[dict, int, int]:
    """Only counts transitions between windows exactly 15 minutes apart --
    see module docstring for why this matters."""
    transitions = {"up_to_up": 0, "up_to_down": 0, "down_to_up": 0, "down_to_down": 0}
    skipped_gaps = 0
    counted = 0

    for i in range(len(outcomes) - 1):
        t1 = datetime.fromisoformat(outcomes[i]["window_id"])
        t2 = datetime.fromisoformat(outcomes[i + 1]["window_id"])
        if (t2 - t1) != timedelta(minutes=15):
            skipped_gaps += 1
            continue
        prev, nxt = outcomes[i]["outcome"], outcomes[i + 1]["outcome"]
        transitions[f"{prev}_to_{nxt}"] += 1
        counted += 1

    return transitions, skipped_gaps, counted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only use the most recent N windows (default: all available)")
    parser.add_argument("--region", type=str, default=REGION)
    args = parser.parse_args()

    log.info("Pulling all window outcomes...")
    outcomes = get_all_windows_with_outcomes(region=args.region)
    if args.limit:
        outcomes = outcomes[-args.limit:]

    n_authoritative = sum(1 for o in outcomes if o["authoritative"])
    log.info(f"Found {len(outcomes)} windows with a usable outcome "
             f"({n_authoritative} from Kalshi's authoritative settlement, "
             f"{len(outcomes) - n_authoritative} from our own approximation)")

    transitions, skipped_gaps, counted = compute_transition_probabilities(outcomes)
    log.info(f"{counted} valid consecutive-window transitions used, "
             f"{skipped_gaps} pairs skipped due to a gap in the data")

    up_total = transitions["up_to_up"] + transitions["up_to_down"]
    down_total = transitions["down_to_up"] + transitions["down_to_down"]

    print(f"\n=== Outcome repeat probability (n={counted} valid consecutive transitions) ===")
    if up_total > 0:
        p_up_after_up = transitions["up_to_up"] / up_total
        print(f"P(up | previous was up):     {p_up_after_up:.3f}  "
              f"({transitions['up_to_up']}/{up_total})")
    else:
        print("P(up | previous was up):     no data")

    if down_total > 0:
        p_down_after_down = transitions["down_to_down"] / down_total
        print(f"P(down | previous was down): {p_down_after_down:.3f}  "
              f"({transitions['down_to_down']}/{down_total})")
    else:
        print("P(down | previous was down): no data")

    print(f"\nFor reference, pure 50/50 randomness would show both at ~0.500.")
    print(f"Consistently above 0.5 on both = momentum (trend continues).")
    print(f"Consistently below 0.5 on both = mean-reversion (trend flips).")


if __name__ == "__main__":
    main()
