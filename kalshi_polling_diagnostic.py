"""
Direct diagnostic against Kalshi's live API -- checks whether consecutive
polls return genuinely fresh data, or whether something (server-side
caching, a stale market ticker, etc.) is causing repeated identical
responses despite the underlying market moving every 1-2 seconds (per
direct visual observation of Kalshi's own UI).

Polls every ~1s for ~30s, printing the raw yes_bid/yes_ask/ticker each
time, plus any HTTP caching-related headers Kalshi's response includes.
If consecutive polls return byte-identical prices for many seconds
straight while the real market is visibly moving, that's strong evidence
of server-side caching upstream of us, not a quiet/thin market.

Usage:
    python3 kalshi_polling_diagnostic.py [--seconds 30] [--interval 1.0]
"""
from __future__ import annotations
import argparse
import time

import requests

KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXBTC15M"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    start = time.time()
    last_values = None
    poll_num = 0

    while time.time() - start < args.seconds:
        poll_num += 1
        t0 = time.time()
        try:
            resp = requests.get(
                f"{KALSHI_BASE_URL}/markets",
                params={"series_ticker": SERIES_TICKER, "status": "open", "limit": 5},
                timeout=5,
            )
            elapsed_ms = (time.time() - t0) * 1000

            cache_headers = {
                k: v for k, v in resp.headers.items()
                if k.lower() in ("cache-control", "age", "x-cache", "cf-cache-status", "etag", "last-modified")
            }

            markets = resp.json().get("markets", [])
            if not markets:
                print(f"[{poll_num:3d}] t+{time.time()-start:5.1f}s  NO OPEN MARKET FOUND")
                time.sleep(max(0, args.interval - (time.time() - t0)))
                continue

            m = markets[0]
            ticker = m.get("ticker")
            yes_bid = m.get("yes_bid_dollars")
            yes_ask = m.get("yes_ask_dollars")
            current = (ticker, yes_bid, yes_ask)

            changed = "CHANGED" if last_values is not None and current != last_values else (
                "same" if last_values is not None else "first"
            )

            print(f"[{poll_num:3d}] t+{time.time()-start:5.1f}s  "
                  f"({elapsed_ms:.0f}ms)  ticker={ticker}  yes_bid={yes_bid}  yes_ask={yes_ask}  "
                  f"[{changed}]"
                  + (f"  cache_headers={cache_headers}" if cache_headers else ""))

            last_values = current

        except Exception as e:
            print(f"[{poll_num:3d}] ERROR: {e}")

        time.sleep(max(0, args.interval - (time.time() - t0)))

    print("\nDone. If many consecutive polls show 'same' despite real-time price movement being "
          "visible on Kalshi's own site, and cache headers show a non-trivial max-age/Age value, "
          "that confirms server-side caching is the cause. If values genuinely stay identical with "
          "NO cache headers present, the issue is more likely elsewhere in our own pipeline "
          "(e.g. the poller silently failing and re-writing a stale in-memory value).")


if __name__ == "__main__":
    main()
