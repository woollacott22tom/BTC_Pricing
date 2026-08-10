"""
Ingestion service. Run this continuously on the EC2 box (systemd or
tmux/screen). Connects to Binance's public WebSocket streams for BTCUSDT:

  - trade stream (price/volume prints)
  - partial book depth stream (top 20 levels, ~100ms updates)

No API key required -- these are public market data streams.

Every ~1 second it:
  1. Computes a feature snapshot from the rolling buffer
  2. Writes a tick row (raw fields + features) to DynamoDB
  3. If this is the FIRST tick of a new window, records the strike (open) price
  4. If this is within the settlement averaging period, buffers the price for TWAP
  5. On window close, computes settlement (60s TWAP), writes the window summary,
     and resets the strike for the new window

Costs to be aware of: this writes ~1 row/sec/table to DynamoDB continuously.
At on-demand pricing this is comfortably within free tier at this volume, but
if you later increase tick frequency, keep an eye on write units.
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
from datetime import datetime, timezone

import websockets

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.window_utils import window_id_for, seconds_remaining, in_settlement_average_period
from features.compute import Tick, RollingBuffer, compute_feature_snapshot
from ingestion.dynamo_client import put_tick, put_window_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingestion")

BINANCE_WS = "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/btcusdt@depth20@100ms"
REGION = os.environ.get("AWS_REGION", "us-east-1")
TICK_LOG_INTERVAL_SEC = 1.0


class WindowState:
    def __init__(self):
        self.window_id: str | None = None
        self.strike_price: float | None = None
        self.settlement_prices: list[float] = []  # collected during final 60s

    def maybe_roll(self, now: datetime, current_price: float):
        """Returns (closed_window_id, closed_summary) if a window just
        closed, else None. Captures the OLD window_id before rolling state
        forward, so callers log the summary under the correct id."""
        wid = window_id_for(now)
        if self.window_id is None:
            self.window_id = wid
            self.strike_price = current_price
            log.info(f"Window opened: {wid} strike={current_price}")
            return None  # no prior window to close

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
        # crude flip proxy: did price cross the strike at any point during
        # the settlement averaging window, when it wasn't going to based on
        # the pre-final-minute trend? Refined properly at training time from
        # full tick history -- this is just a fast live flag.
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
    last_depth: dict | None = None

    async for ws in _reconnecting_ws(BINANCE_WS):
        try:
            async for raw in ws:
                msg = json.loads(raw)
                stream = msg.get("stream", "")
                data = msg.get("data", {})
                now = datetime.now(timezone.utc)
                now_ts = now.timestamp()

                if stream.endswith("@trade"):
                    price = float(data["p"])
                    volume = float(data["q"])
                    bid_depth = last_depth["bid"] if last_depth else None
                    ask_depth = last_depth["ask"] if last_depth else None
                    best_bid = last_depth["best_bid"] if last_depth else None
                    best_ask = last_depth["best_ask"] if last_depth else None

                    tick = Tick(
                        ts=now_ts, price=price, volume=volume,
                        best_bid=best_bid, best_ask=best_ask,
                        bid_depth_top10=bid_depth, ask_depth_top10=ask_depth,
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
                                "best_bid": best_bid, "best_ask": best_ask,
                                "bid_depth_top10": bid_depth, "ask_depth_top10": ask_depth,
                            },
                            features=feats,
                            region_name=REGION,
                        )
                        last_logged = now_ts

                elif stream.endswith("@depth20@100ms"):
                    bids = data.get("bids", [])[:10]
                    asks = data.get("asks", [])[:10]
                    if bids and asks:
                        last_depth = {
                            "bid": sum(float(sz) for _, sz in bids),
                            "ask": sum(float(sz) for _, sz in asks),
                            "best_bid": float(bids[0][0]),
                            "best_ask": float(asks[0][0]),
                        }
        except websockets.ConnectionClosed:
            log.warning("WebSocket closed, reconnecting...")
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
