"""
Polls Kalshi's public REST API for the currently-open BTC 15-min market's
live price, alongside your existing Coinbase feed -- lets you empirically
test whether Coinbase price action leads Kalshi's contract price, and by
how much (rather than trusting the visual impression alone).

No API key needed -- Kalshi's market data endpoints are public. Confirmed
base URL from Kalshi's own docs: https://external-api.kalshi.com/trade-api/v2

Rather than parsing/reconstructing Kalshi's date-suffixed ticker ourselves
(e.g. "KXBTC15M-26AUG112115"), this asks Kalshi which market in the
KXBTC15M series is currently open, and pairs that response with the same
window_id your own window_utils.py computes -- both systems' 15-minute
windows align to the same UTC :00/:15/:30/:45 boundaries by construction.

Kalshi prices are in cents (1-99), which map directly to implied
probability (e.g. yes_bid=62 means the market implies ~62% probability of
"yes"/up). We store yes_bid, yes_ask, and their midpoint.

Output: writes rows to a new DynamoDB table, kalshi_prices, keyed the same
way as btc_ticks (window_id + timestamp), so the two can be joined later
for lead-lag analysis.

Usage (run continuously, e.g. as its own systemd service):
    python3 kalshi_poller.py
"""
from __future__ import annotations
import logging
import os
import time
from datetime import datetime, timezone
from decimal import Decimal

import requests
import boto3

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from window_utils import window_id_for

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kalshi_poller")

KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXBTC15M"
POLL_INTERVAL_SEC = 2.0  # well under Kalshi's public rate limit (~30 req/sec)
REGION = os.environ.get("AWS_REGION", "us-east-1")
TABLE_NAME = "kalshi_prices"


def fetch_current_market() -> dict | None:
    """Returns the currently open KXBTC15M market's data, or None if the
    request fails or no open market is found (e.g. between windows)."""
    try:
        resp = requests.get(
            f"{KALSHI_BASE_URL}/markets",
            params={"series_ticker": SERIES_TICKER, "status": "open", "limit": 5},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        log.warning(f"Kalshi request failed: {e}")
        return None

    markets = data.get("markets", [])
    if not markets:
        log.warning("No open KXBTC15M market found in response")
        return None
    return markets[0]


def extract_prices(market: dict) -> dict | None:
    """Kalshi's live API returns yes_bid_dollars/yes_ask_dollars (floats,
    already in probability form e.g. 0.62), not the cents-integer fields
    ('yes_bid'/'yes_ask') shown in some doc examples. Checks the dollars
    fields first since that's what the live endpoint actually returns,
    with a fallback to the cents-style fields in case that ever changes."""
    yes_bid_dollars = market.get("yes_bid_dollars")
    yes_ask_dollars = market.get("yes_ask_dollars")

    if yes_bid_dollars is not None and yes_ask_dollars is not None:
        yes_bid_cents = float(yes_bid_dollars) * 100.0
        yes_ask_cents = float(yes_ask_dollars) * 100.0
    else:
        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")
        if yes_bid is None or yes_ask is None:
            log.warning(f"Unexpected market response shape, no recognized price fields: {list(market.keys())}")
            return None
        yes_bid_cents = float(yes_bid)
        yes_ask_cents = float(yes_ask)
        # guard against a dollars-format value slipping through here too
        if 0 < yes_bid_cents < 1:
            yes_bid_cents *= 100
        if 0 < yes_ask_cents < 1:
            yes_ask_cents *= 100

    return {
        "yes_bid_cents": yes_bid_cents,
        "yes_ask_cents": yes_ask_cents,
        "yes_mid_cents": (yes_bid_cents + yes_ask_cents) / 2.0,
        "ticker": market.get("ticker"),
        "volume": float(market.get("volume_fp", market.get("volume", 0)) or 0),
    }


def put_kalshi_price(window_id: str, timestamp: float, prices: dict, region: str = REGION):
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(TABLE_NAME)
    item = {
        "window_id": window_id,
        "timestamp": Decimal(str(timestamp)),
        "yes_bid_cents": Decimal(str(prices["yes_bid_cents"])),
        "yes_ask_cents": Decimal(str(prices["yes_ask_cents"])),
        "yes_mid_cents": Decimal(str(prices["yes_mid_cents"])),
        "kalshi_ticker": prices["ticker"] or "unknown",
        "volume": Decimal(str(prices["volume"])),
    }
    table.put_item(Item=item)


def run():
    log.info(f"Starting Kalshi poller for series {SERIES_TICKER}, polling every {POLL_INTERVAL_SEC}s")
    last_ticker_seen = None

    while True:
        loop_start = time.time()
        now = datetime.now(timezone.utc)

        market = fetch_current_market()
        if market:
            prices = extract_prices(market)
            if prices:
                wid = window_id_for(now)
                put_kalshi_price(wid, now.timestamp(), prices, region=REGION)

                if prices["ticker"] != last_ticker_seen:
                    log.info(f"Tracking market {prices['ticker']} (window {wid}) "
                             f"yes_bid={prices['yes_bid_cents']}c yes_ask={prices['yes_ask_cents']}c")
                    last_ticker_seen = prices["ticker"]

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, POLL_INTERVAL_SEC - elapsed))


if __name__ == "__main__":
    run()
