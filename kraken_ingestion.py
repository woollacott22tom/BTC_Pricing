"""
Ingestion service for Kraken -- mirrors run_ingestion.py's Coinbase pipeline
structurally (same Tick/RollingBuffer/compute_feature_snapshot from
compute.py, fully exchange-agnostic), but writes to a SEPARATE table
(kraken_ticks) so it can never risk the stable, already-working Coinbase
pipeline.

Endpoint: wss://ws.kraken.com/v2 -- public, no auth required for "trade"
and "book" channels (Kraken's level3/order-level channel needs a token;
the aggregated level2 "book" channel used here does not, confirmed from
Kraken's own docs).

Every ~1 second it:
  1. Computes a feature snapshot from the rolling buffer + live order book
  2. Writes a tick row to kraken_ticks
  3. Tracks window open/settlement the same way as Coinbase, purely for
     cross-exchange comparison purposes -- Kraken doesn't have its own
     concept of a 15-min settlement window, we're just aligning to the
     same boundaries as everything else in this project.

Usage:
    python3 kraken_ingestion.py
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
from kraken_order_book import KrakenOrderBook
from kraken_dynamo_client import put_kraken_tick

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kraken_ingestion")

KRAKEN_WS = "wss://ws.kraken.com/v2"
SYMBOL = "BTC/USD"
REGION = os.environ.get("AWS_REGION", "us-east-1")
TICK_LOG_INTERVAL_SEC = 1.0


KRAKEN_BOOK_DEPTH = 100  # explicit depth request -- was previously unset,
                          # which likely defaulted to a shallow depth
                          # (commonly 10), silently capping imbalance_20
                          # and imbalance_50 to the same shallow view


def _subscribe_msg(channel: str) -> dict:
    params = {"channel": channel, "symbol": [SYMBOL]}
    if channel == "book":
        params["depth"] = KRAKEN_BOOK_DEPTH
    return {"method": "subscribe", "params": params}


async def run():
    buf = RollingBuffer(max_seconds=900)
    strike_price: float | None = None
    current_window_id: str | None = None
    last_logged = 0.0
    last_trade_price: float | None = None
    pending_volume = 0.0

    # Same trade-flow diagnostic added to crypto_ingestion.py, for a fair
    # apples-to-apples comparison of actual trade frequency between feeds.
    trade_message_count = 0
    trade_entry_count = 0
    last_diag_log = time.time()

    backoff = 1.0
    while True:
        try:
            async with websockets.connect(KRAKEN_WS, ping_interval=20, ping_timeout=20, max_size=None) as ws:
                backoff = 1.0
                await ws.send(json.dumps(_subscribe_msg("trade")))
                await ws.send(json.dumps(_subscribe_msg("book")))
                book = KrakenOrderBook()
                msg_count = 0

                async for raw in ws:
                    msg_count += 1
                    if msg_count <= 5:
                        log.info(f"Raw message #{msg_count}: {raw[:300]}")

                    if time.time() - last_diag_log >= 30.0:
                        log.info(f"[trade-flow diagnostic] {trade_message_count} trade messages, "
                                 f"{trade_entry_count} individual trade entries in the last 30s")
                        trade_message_count = 0
                        trade_entry_count = 0
                        last_diag_log = time.time()

                    msg = json.loads(raw)
                    channel = msg.get("channel")
                    now = datetime.now(timezone.utc)
                    now_ts = now.timestamp()

                    if channel == "book":
                        msg_type = msg.get("type")
                        for entry in msg.get("data", []):
                            book.apply_message(msg_type, entry)

                    elif channel == "trade":
                        trade_message_count += 1
                        for entry in msg.get("data", []):
                            trade_entry_count += 1
                            try:
                                price = float(entry["price"])
                                volume = float(entry["qty"])
                            except (KeyError, TypeError, ValueError):
                                continue
                            last_trade_price = price
                            pending_volume += volume

                    elif msg.get("method") == "subscribe" and not msg.get("success", True):
                        log.error(f"Kraken subscribe error: {msg}")

                    # Fires on EVERY message (book or trade), not just trade
                    # arrival -- Kraken's book updates independently of
                    # trades; gating on trade arrival meant book-only
                    # movement was invisible to the stored dataset. Also
                    # fixes volume from multiple trades arriving between
                    # log points being silently dropped instead of summed.
                    if last_trade_price is not None and now_ts - last_logged >= TICK_LOG_INTERVAL_SEC:
                        tick = Tick(
                            ts=now_ts, price=last_trade_price, volume=pending_volume,
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
                            strike_price = last_trade_price
                            log.info(f"Window rolled: {wid} strike={last_trade_price}")

                        feats = compute_feature_snapshot(buf, strike_price, now_ts)
                        feats["seconds_remaining"] = seconds_remaining(now, current_window_id)
                        put_kraken_tick(
                            window_id=current_window_id,
                            timestamp=now_ts,
                            tick_fields={
                                "price": last_trade_price, "volume": pending_volume,
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
                        pending_volume = 0.0

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
