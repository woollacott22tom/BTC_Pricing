"""
Ingestion service for Crypto.com Exchange -- same feature set as Coinbase
and Kraken (via compute.py, fully exchange-agnostic), writing to its own
table (cryptocom_ticks) so it can never risk any other pipeline.

IMPORTANT CAVEAT: unlike Kraken's integration, this was built without being
able to verify against Crypto.com's actual live message shape (their docs
site is JS-rendered, didn't return real content when fetched). Only the
channel naming convention was confirmed from search results:
  - "book.{instrument_name}.{depth}" e.g. "book.BTC_USD.50"
  - "trade.{instrument_name}" e.g. "trade.BTC_USD"

Endpoint: wss://stream.crypto.com/exchange/v1/market -- this URL is my
best understanding from what's broadly documented for Crypto.com's public
Market Data websocket, NOT independently verified the way Kraken's was.
If this doesn't connect, that's the first thing to check.

Crypto.com's API is known to require responding to server-sent heartbeat
pings (method: "public/heartbeat") with a matching respond-heartbeat
message, or the connection gets dropped -- handled below.

Buy/sell volume: Crypto.com's own public changelog explicitly clarifies
"Side is the side of the taker order" for the trade channel -- confirmed
directly, no inversion needed (unlike Coinbase). Given the same shape
uncertainty flagged above, side parsing is defensive here too, trying
both the casing seen in Crypto.com's older docs (BUY/SELL) and the
lowercase convention used elsewhere (buy/sell) -- an unrecognized value
is left unattributed rather than guessed at.

The raw-message debug logging (first 10 messages) is deliberately more
generous here than Kraken's was, since we genuinely don't know the shape
yet -- use these logs to correct crypto_order_book.py's parsing if needed.

Usage:
    python3 crypto_ingestion.py
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

import websockets

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flat5"))

from window_utils import window_id_for, seconds_remaining
from compute import Tick, RollingBuffer, compute_feature_snapshot
from crypto_order_book import CryptoComOrderBook
from crypto_dynamo_client import put_cryptocom_tick

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cryptocom_ingestion")

CRYPTOCOM_WS = "wss://stream.crypto.com/exchange/v1/market"
INSTRUMENT = "BTC_USD"
BOOK_DEPTH = 50
REGION = os.environ.get("AWS_REGION", "us-east-1")
TICK_LOG_INTERVAL_SEC = 1.0


def _subscribe_msg() -> dict:
    return {
        "id": 1,
        "method": "subscribe",
        "params": {
            "channels": [f"book.{INSTRUMENT}.{BOOK_DEPTH}", f"trade.{INSTRUMENT}"],
        },
    }


def _try_parse_trade(entry: dict) -> tuple[float, float, str | None] | None:
    """Defensive parsing -- Crypto.com's docs have shown both short (p/q)
    and long (price/quantity) field name conventions across versions.
    Now also extracts taker side (confirmed via Crypto.com's own changelog:
    "Side is the side of the taker order", no inversion needed) -- tried
    against both the uppercase (BUY/SELL) casing seen in their older docs
    and lowercase, since the live shape hasn't been independently verified
    the way Kraken's was. An unrecognized/missing side returns None for
    that element rather than guessing."""
    try:
        price = entry.get("p", entry.get("price"))
        qty = entry.get("q", entry.get("quantity", entry.get("qty")))
        if price is None or qty is None:
            return None
        side_raw = entry.get("s", entry.get("side"))
        side = None
        if isinstance(side_raw, str):
            if side_raw.upper() == "BUY":
                side = "BUY"
            elif side_raw.upper() == "SELL":
                side = "SELL"
        return float(price), float(qty), side
    except (TypeError, ValueError):
        return None


async def run():
    buf = RollingBuffer(max_seconds=900)
    strike_price: float | None = None
    current_window_id: str | None = None
    last_logged = 0.0

    # Ongoing trade-flow diagnostic -- the initial raw-message debug
    # logging (first 10 messages) gets exhausted by book-channel traffic
    # within seconds, giving no visibility into trade frequency after
    # that. This reports actual trade throughput every 30s regardless.
    trade_message_count = 0
    trade_entry_count = 0
    last_diag_log = time.time()

    backoff = 1.0
    while True:
        try:
            async with websockets.connect(CRYPTOCOM_WS, ping_interval=20, ping_timeout=20, max_size=None) as ws:
                backoff = 1.0
                await asyncio.sleep(1.0)  # Crypto.com recommends a 1s pause before subscribing
                await ws.send(json.dumps(_subscribe_msg()))
                book = CryptoComOrderBook()
                msg_count = 0

                async for raw in ws:
                    msg_count += 1
                    if msg_count <= 10:
                        log.info(f"Raw message #{msg_count}: {raw[:400]}")

                    if time.time() - last_diag_log >= 30.0:
                        log.info(f"[trade-flow diagnostic] {trade_message_count} trade messages, "
                                 f"{trade_entry_count} individual trade entries in the last 30s")
                        trade_message_count = 0
                        trade_entry_count = 0
                        last_diag_log = time.time()

                    msg = json.loads(raw)

                    if msg.get("method") == "public/heartbeat":
                        await ws.send(json.dumps({"id": msg.get("id"), "method": "public/respond-heartbeat"}))
                        continue

                    result = msg.get("result", {})
                    channel = result.get("channel") or msg.get("channel")
                    now = datetime.now(timezone.utc)
                    now_ts = now.timestamp()

                    if channel == "book":
                        for entry in result.get("data", []):
                            # Crypto.com sends a FULL replacement snapshot on
                            # every push (confirmed from live data), not
                            # incremental deltas -- always clear + repopulate
                            book.apply_message(entry, is_snapshot=True)

                    elif channel == "trade":
                        trade_message_count += 1
                        for entry in result.get("data", []):
                            trade_entry_count += 1
                            parsed = _try_parse_trade(entry)
                            if parsed is None:
                                log.warning(f"Trade entry failed to parse -- unrecognized shape: {entry}")
                                continue
                            price, volume, side = parsed

                            # No cross-trade volume accumulator in this
                            # file (unlike Kraken/Coinbase) -- each trade
                            # writes its own tick immediately (rate-limited
                            # below), so buy/sell volume here is just this
                            # SINGLE trade's own volume attributed to
                            # whichever side it was, consistent with how
                            # "volume" itself already works in this file.
                            buy_volume = volume if side == "BUY" else 0.0
                            sell_volume = volume if side == "SELL" else 0.0

                            tick = Tick(
                                ts=now_ts, price=price, volume=volume,
                                best_bid=book.best_bid(), best_ask=book.best_ask(),
                                bid_depth_top10=book.bid_depth(10), ask_depth_top10=book.ask_depth(10),
                                bid_depth_5=book.bid_depth(5), ask_depth_5=book.ask_depth(5),
                                bid_depth_20=book.bid_depth(20), ask_depth_20=book.ask_depth(20),
                                bid_depth_50=book.bid_depth(50), ask_depth_50=book.ask_depth(50),
                            )
                            buf.add(tick)

                            wid = window_id_for(now)
                            if current_window_id != wid:
                                current_window_id = wid
                                strike_price = price
                                log.info(f"Window rolled: {wid} strike={price}")

                            if now_ts - last_logged >= TICK_LOG_INTERVAL_SEC:
                                feats = compute_feature_snapshot(buf, strike_price, now_ts)
                                feats["seconds_remaining"] = seconds_remaining(now, current_window_id)
                                put_cryptocom_tick(
                                    window_id=current_window_id,
                                    timestamp=now_ts,
                                    tick_fields={
                                        "price": price, "volume": volume,
                                        "buy_volume": buy_volume, "sell_volume": sell_volume,
                                        "best_bid": tick.best_bid, "best_ask": tick.best_ask,
                                        "bid_depth_top10": tick.bid_depth_top10,
                                        "ask_depth_top10": tick.ask_depth_top10,
                                        "bid_depth_5": tick.bid_depth_5, "ask_depth_5": tick.ask_depth_5,
                                        "bid_depth_20": tick.bid_depth_20, "ask_depth_20": tick.ask_depth_20,
                                        "bid_depth_50": tick.bid_depth_50, "ask_depth_50": tick.ask_depth_50,
                                    },
                                    features=feats,
                                    region_name=REGION,
                                )
                                last_logged = now_ts

                    elif msg.get("code") not in (0, None):
                        log.error(f"Crypto.com error response: {msg}")

        except websockets.ConnectionClosed as e:
            log.warning(f"WebSocket closed, reconnecting... code={e.code} reason={e.reason!r}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        except Exception as e:
            log.error(f"Connection error: {e}, retrying in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


if __name__ == "__main__":
    asyncio.run(run())
