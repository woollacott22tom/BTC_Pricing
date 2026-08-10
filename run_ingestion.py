"""
Ingestion service. Run this continuously on the EC2 box (systemd or
tmux/screen). Connects to Coinbase's Advanced Trade WebSocket API for
BTC-USD.

  No API key needed: per Coinbase's current docs, most Advanced Trade
  market data channels -- including "level2" (full order book) and
  "market_trades" -- are now public and don't require authentication.
  (An earlier version of this project went down a JWT/API-key path based
  on the OLDER Exchange product's auth requirement, which doesn't apply
  here. Advanced Trade's market data is simply public.)

  - "market_trades" channel: trade prints (price/size)
  - "level2" channel: full order book, snapshot then incremental updates
    (tracked locally via order_book.OrderBook)

Every ~1 second it:
  1. Computes a feature snapshot from the rolling buffer + live order book
  2. Writes a tick row (raw fields + features) to DynamoDB
  3. If this is the FIRST tick of a new window, records the strike (open) price
  4. If this is within the settlement averaging period, buffers the price for TWAP
  5. On window close, computes settlement (60s TWAP), writes the window summary,
     and resets the strike for the new window
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

import websockets

from window_utils import window_id_for, seconds_remaining, in_settlement_average_period
from compute import Tick, RollingBuffer, compute_feature_snapshot
from dynamo_client import put_tick, put_window_summary
from order_book import OrderBook

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingestion")

ADVANCED_TRADE_WS = "wss://advanced-trade-ws.coinbase.com"
PRODUCT_ID = "BTC-USD"
REGION = os.environ.get("AWS_REGION", "us-east-1")
TICK_LOG_INTERVAL_SEC = 1.0


def _subscribe_msg(channel: str) -> dict:
    # No "jwt" field -- these channels are public. Coinbase disconnects
    # you after 5s if no subscribe message is sent, but no auth is needed.
    return {
        "type": "subscribe",
        "product_ids": [PRODUCT_ID],
        "channel": channel,
    }


class WindowState:
    def __init__(self):
        self.window_id: str | None = None
        self.strike_price: float | None = None
        self.settlement_prices: list[float] = []

    def maybe_roll(self, now: datetime, current_price: float):
        wid = window_id_for(now)
        if self.window_id is None:
            self.window_id = wid
            self.strike_price = current_price
            log.info(f"Window opened: {wid} strike={current_price}")
            return None

        if wid != self.window_id:
            old_window_id = self.window_id
            closed_summary = self._close(current_price)
            self.window_id = wid
            self.strike_price = current_price
            self.settlement_prices = []
            log.info(f"Window opened: {wid} strike={current_price}")
            return old_window_id, closed_summary
        return None

    def record_settlement_tick(self, now: datetime, price: float):
        if in_settlement_average_period(now, self.window_id):
            self.settlement_prices.append(price)

    def _close(self, fallback_price: float) -> dict:
        settlement_price = (
            sum(self.settlement_prices) / len(self.settlement_prices)
            if self.settlement_prices
            else fallback_price
        )
        outcome = "up" if settlement_price >= self.strike_price else "down"
        max_dev = max(
            (abs(p - self.strike_price) for p in self.settlement_prices),
            default=0.0,
        )
        flip_occurred = len(set(p >= self.strike_price for p in self.settlement_prices)) > 1
        return {
            "open_price": self.strike_price,
            "settlement_price": settlement_price,
            "outcome": outcome,
            "max_deviation_final_60s": max_dev,
            "flip_occurred": flip_occurred,
            "closed_at": time.time(),
        }


async def run():
    buf = RollingBuffer(max_seconds=900)
    state = WindowState()
    last_logged = 0.0

    async for ws in _reconnecting_ws(ADVANCED_TRADE_WS):
        try:
            await ws.send(json.dumps(_subscribe_msg("market_trades")))
            await ws.send(json.dumps(_subscribe_msg("level2")))
            book = OrderBook()

            async for raw in ws:
                msg = json.loads(raw)
                channel = msg.get("channel")
                now = datetime.now(timezone.utc)
                now_ts = now.timestamp()

                if channel == "l2_data":
                    for event in msg.get("events", []):
                        book.apply_event(event)

                elif channel == "market_trades":
                    for event in msg.get("events", []):
                        if event.get("type") != "update":
                            continue  # skip the initial backfill snapshot of past trades
                        for trade in event.get("trades", []):
                            try:
                                price = float(trade["price"])
                                volume = float(trade["size"])
                            except (KeyError, TypeError, ValueError):
                                continue

                            tick = Tick(
                                ts=now_ts, price=price, volume=volume,
                                best_bid=book.best_bid(), best_ask=book.best_ask(),
                                bid_depth_top10=book.top10_bid_depth(),
                                ask_depth_top10=book.top10_ask_depth(),
                            )
                            buf.add(tick)

                            rolled = state.maybe_roll(now, price)
                            if rolled:
                                closed_window_id, closed_summary = rolled
                                put_window_summary(closed_window_id, closed_summary, region_name=REGION)
                                log.info(f"Window closed: outcome={closed_summary['outcome']} flip={closed_summary['flip_occurred']}")

                            state.record_settlement_tick(now, price)

                            if now_ts - last_logged >= TICK_LOG_INTERVAL_SEC:
                                feats = compute_feature_snapshot(buf, state.strike_price, now_ts)
                                feats["seconds_remaining"] = seconds_remaining(now, state.window_id)
                                put_tick(
                                    window_id=state.window_id,
                                    timestamp=now_ts,
                                    tick_fields={
                                        "price": price, "volume": volume,
                                        "best_bid": tick.best_bid, "best_ask": tick.best_ask,
                                        "bid_depth_top10": tick.bid_depth_top10,
                                        "ask_depth_top10": tick.ask_depth_top10,
                                    },
                                    features=feats,
                                    region_name=REGION,
                                )
                                last_logged = now_ts

                elif msg.get("type") == "error":
                    log.error(f"Coinbase feed error: {msg}")

        except websockets.ConnectionClosed as e:
            log.warning(f"WebSocket closed, reconnecting... code={e.code} reason={e.reason!r}")
            continue


async def _reconnecting_ws(url: str, backoff_start: float = 1.0, backoff_max: float = 30.0):
    backoff = backoff_start
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                backoff = backoff_start
                yield ws
        except Exception as e:
            log.error(f"WS connect failed: {e}, retrying in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, backoff_max)


if __name__ == "__main__":
    asyncio.run(run())
