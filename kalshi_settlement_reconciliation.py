"""
Pulls Kalshi's OWN authoritative settlement result for closed KXBTC15M
markets and writes it onto the corresponding btc_windows row, alongside
our own Coinbase-TWAP-approximated outcome.

Kalshi's public GET /markets?status=settled response includes:
  - "result": "yes"/"no" -- the true, official outcome
  - "expiration_value": the actual CF Benchmarks BRTI settlement price
    used to determine that outcome (confirmed by cross-referencing real
    settled markets: each window's expiration_value exactly equals the
    NEXT window's floor_strike, matching the rules text describing each
    window's strike as the prior window's 60s-average settlement)
  - "floor_strike": the strike/target used for THIS window
  - "open_time": lets us compute the matching window_id

This is the authoritative source -- our own settlement calculation
(Coinbase-only 60s TWAP) is an APPROXIMATION of what this real value
turns out to be, since BRTI blends multiple exchanges' order-book data,
not just Coinbase trades. This script also reports how often our own
approximation disagreed with Kalshi's real outcome, to make the actual
scope of any divergence visible rather than assumed.

Usage:
    python3 kalshi_settlement_reconciliation.py [--limit 200]
"""
from __future__ import annotations
import argparse
import logging
from datetime import datetime, timezone

import requests
import boto3
from boto3.dynamodb.conditions import Key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kalshi_settlement_reconciliation")

KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXBTC15M"
WINDOWS_TABLE = "btc_windows"


def fetch_settled_markets(limit: int) -> list[dict]:
    """Paginates through settled KXBTC15M markets, same pattern as
    backfill_kalshi.py's approach (already validated tonight)."""
    markets = []
    cursor = None
    while len(markets) < limit:
        params = {"series_ticker": SERIES_TICKER, "status": "settled", "limit": min(200, limit - len(markets))}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{KALSHI_BASE_URL}/markets", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("markets", [])
        if not batch:
            break
        markets.extend(batch)
        cursor = data.get("cursor")
        if not cursor:
            break
    return markets[:limit]


def market_to_window_id(market: dict) -> str | None:
    """Converts Kalshi's open_time into our own window_id format
    (matches window_utils.window_id_for()'s output exactly)."""
    open_time_str = market.get("open_time")
    if not open_time_str:
        return None
    try:
        dt = datetime.fromisoformat(open_time_str.replace("Z", "+00:00"))
        return dt.isoformat()
    except ValueError:
        return None


def parse_settlement(market: dict) -> dict | None:
    """Defensive parsing -- result is a plain lowercase string, but
    expiration_value/floor_strike may arrive as either a string or a raw
    number depending on the field (confirmed different in real captured
    responses: expiration_value is quoted, floor_strike is not)."""
    try:
        result = market.get("result")
        if result not in ("yes", "no"):
            return None
        expiration_value = market.get("expiration_value")
        floor_strike = market.get("floor_strike")
        if expiration_value is None or floor_strike is None:
            return None
        return {
            "kalshi_true_outcome": "up" if result == "yes" else "down",
            "kalshi_true_settlement_price": float(str(expiration_value)),
            "kalshi_true_strike": float(str(floor_strike)),
            "ticker": market.get("ticker"),
        }
    except (TypeError, ValueError):
        return None


def get_existing_window(table, window_id: str) -> dict | None:
    resp = table.get_item(Key={"window_id": window_id})
    return resp.get("Item")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--dry-run", action="store_true", help="Only report, don't write to DynamoDB")
    args = parser.parse_args()

    log.info(f"Fetching up to {args.limit} settled {SERIES_TICKER} markets...")
    markets = fetch_settled_markets(args.limit)
    log.info(f"Found {len(markets)} settled markets")

    dynamodb = boto3.resource("dynamodb", region_name=args.region)
    table = dynamodb.Table(WINDOWS_TABLE)

    reconciled = 0
    no_matching_window = 0
    agreements = 0
    disagreements = 0
    disagreement_details = []

    for market in markets:
        settlement = parse_settlement(market)
        if settlement is None:
            continue

        window_id = market_to_window_id(market)
        if window_id is None:
            continue

        existing = get_existing_window(table, window_id)
        if existing is None:
            no_matching_window += 1
            continue

        our_outcome = existing.get("outcome")
        kalshi_outcome = settlement["kalshi_true_outcome"]

        if our_outcome is not None:
            if our_outcome == kalshi_outcome:
                agreements += 1
            else:
                disagreements += 1
                disagreement_details.append({
                    "window_id": window_id,
                    "ticker": settlement["ticker"],
                    "our_outcome": our_outcome,
                    "kalshi_outcome": kalshi_outcome,
                    "our_settlement_price": existing.get("settlement_price"),
                    "kalshi_settlement_price": settlement["kalshi_true_settlement_price"],
                })

        if not args.dry_run:
            table.update_item(
                Key={"window_id": window_id},
                UpdateExpression=(
                    "SET kalshi_true_outcome = :o, kalshi_true_settlement_price = :p, "
                    "kalshi_true_strike = :s"
                ),
                ExpressionAttributeValues={
                    ":o": kalshi_outcome,
                    ":p": str(settlement["kalshi_true_settlement_price"]),
                    ":s": str(settlement["kalshi_true_strike"]),
                },
            )
        reconciled += 1

    log.info(f"\nReconciled {reconciled} windows against Kalshi's authoritative settlement data")
    log.info(f"{no_matching_window} settled markets had no corresponding row in {WINDOWS_TABLE} "
              f"(likely windows before ingestion started, or a gap)")

    if agreements + disagreements > 0:
        print(f"\n=== Outcome agreement: our approximation vs Kalshi's real result ===")
        print(f"  Agreed:    {agreements} / {agreements + disagreements} "
              f"({100*agreements/(agreements+disagreements):.1f}%)")
        print(f"  Disagreed: {disagreements} / {agreements + disagreements} "
              f"({100*disagreements/(agreements+disagreements):.1f}%)")

        if disagreement_details:
            print(f"\nDisagreement details:")
            for d in disagreement_details:
                print(f"  {d['window_id']} ({d['ticker']}): "
                      f"we said {d['our_outcome']}, Kalshi says {d['kalshi_outcome']} -- "
                      f"our_settlement=${d['our_settlement_price']}, "
                      f"kalshi_settlement=${d['kalshi_settlement_price']}")

    if args.dry_run:
        print("\n(--dry-run: no changes were written to DynamoDB)")


if __name__ == "__main__":
    main()
