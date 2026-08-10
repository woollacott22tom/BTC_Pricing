"""
Backfills historical BTC-USD windows from Coinbase's public REST candle
endpoint, to bootstrap a baseline directional model before enough live tick
data has accumulated.

IMPORTANT LIMITATION: candles only give open/high/low/close/volume per
window -- none of the order-flow features (book imbalance, spread, depth,
momentum sub-features) exist retroactively. This produces a SEPARATE,
simpler dataset/model from the live tick-based pipeline, not a merge.

Coinbase's public candle endpoint:
  GET https://api.exchange.coinbase.com/products/BTC-USD/candles
  params: start, end (ISO8601), granularity (seconds; 900 = 15 min, matches
  Kalshi's window exactly)

Rate limit: public endpoint allows ~10 requests/second; endpoint also caps
each response at 300 candles, so a 6-month backfill requires many paginated
requests (6 months * 30 days * 96 windows/day = ~17,280 candles => ~58 requests
at 300/request). This script paginates automatically with a small delay
between requests to stay well under the rate limit.

Output: writes rows to DynamoDB's btc_windows table, same schema as live
ingestion's window summaries, but with a "source": "backfill" tag so you can
distinguish historical windows from live-collected ones at training time
(their feature availability differs).

Usage:
    python3 backfill_candles.py --months 6
"""
from __future__ import annotations
import argparse
import time
from datetime import datetime, timedelta, timezone

import requests
import boto3
from decimal import Decimal

COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
GRANULARITY_SECONDS = 900  # 15 minutes, matches Kalshi's window exactly
MAX_CANDLES_PER_REQUEST = 300
REQUEST_DELAY_SEC = 0.15  # stay comfortably under the ~10 req/sec public rate limit

WINDOWS_TABLE = "btc_windows"


def fetch_candles(start: datetime, end: datetime, region: str = "us-east-1") -> list[dict]:
    """Fetches all candles in [start, end), paginating as needed."""
    all_candles = []
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(
            chunk_start + timedelta(seconds=GRANULARITY_SECONDS * MAX_CANDLES_PER_REQUEST),
            end,
        )
        params = {
            "start": chunk_start.isoformat(),
            "end": chunk_end.isoformat(),
            "granularity": GRANULARITY_SECONDS,
        }
        resp = requests.get(COINBASE_CANDLES_URL, params=params, timeout=10)
        resp.raise_for_status()
        candles = resp.json()  # each: [time, low, high, open, close, volume]
        all_candles.extend(candles)

        print(f"Fetched {len(candles)} candles for {chunk_start.date()} to {chunk_end.date()}")
        chunk_start = chunk_end
        time.sleep(REQUEST_DELAY_SEC)

    return all_candles


def candle_to_window_summary(candle: list) -> dict:
    """Coinbase candle format: [unix_time, low, high, open, close, volume]"""
    unix_time, low, high, open_price, close_price, volume = candle
    window_open = datetime.fromtimestamp(unix_time, tz=timezone.utc)
    window_id = window_open.isoformat()

    outcome = "up" if close_price >= open_price else "down"
    max_dev = max(abs(high - open_price), abs(low - open_price))

    return {
        "window_id": window_id,
        "open_price": Decimal(str(open_price)),
        "settlement_price": Decimal(str(close_price)),  # approximation -- real settlement
                                                           # is a 60s TWAP, candle close is
                                                           # a single tick, expect some label noise
        "outcome": outcome,
        "max_deviation_final_60s": Decimal(str(max_dev)),  # approximation, not the real final-60s figure
        "flip_occurred": None,  # cannot be determined from candle data, not logged
        "high": Decimal(str(high)),
        "low": Decimal(str(low)),
        "volume": Decimal(str(volume)),
        "closed_at": Decimal(str(unix_time + GRANULARITY_SECONDS)),
        "source": "backfill",
    }


def write_windows(windows: list[dict], region: str = "us-east-1"):
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(WINDOWS_TABLE)

    written = 0
    with table.batch_writer(overwrite_by_pkeys=["window_id"]) as batch:
        for w in windows:
            batch.put_item(Item=w)
            written += 1
            if written % 500 == 0:
                print(f"  ...{written} windows written so far")

    print(f"Done. Wrote {written} backfilled windows to {WINDOWS_TABLE}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=6, help="How many months back to backfill")
    parser.add_argument("--region", type=str, default="us-east-1")
    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30 * args.months)

    print(f"Backfilling BTC-USD 15-min windows from {start.date()} to {end.date()} ({args.months} months)")
    candles = fetch_candles(start, end, region=args.region)
    print(f"\nTotal candles fetched: {len(candles)}")

    windows = [candle_to_window_summary(c) for c in candles]
    write_windows(windows, region=args.region)


if __name__ == "__main__":
    main()
