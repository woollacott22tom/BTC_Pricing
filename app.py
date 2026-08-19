"""
Live scoring API. Runs on the same EC2 box as ingestion (or a second
process) -- maintains its own rolling buffer + order book from the same
public Coinbase Advanced Trade feed (no API key needed -- see
run_ingestion.py's docstring), loads trained model artifacts, and exposes a
small JSON endpoint your GitHub-hosted dashboard can poll.

    uvicorn app:app --host 0.0.0.0 --port 8000

CORS is left open (`*`) since the dashboard is a static site on a different
origin (GitHub Pages). Tighten `allow_origins` to your actual dashboard
domain once you know it.

Endpoints:
  GET /live       -> current window's live directional + flip scores
  GET /health     -> liveness check
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

import websockets
import requests
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from window_utils import window_id_for, seconds_remaining
from compute import Tick, RollingBuffer, compute_feature_snapshot
from order_book import OrderBook
from jwt_auth import build_ws_jwt
from kraken_order_book import KrakenOrderBook
from crypto_order_book import CryptoComOrderBook

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serving")

ADVANCED_TRADE_WS = "wss://advanced-trade-ws.coinbase.com"
PRODUCT_ID = "BTC-USD"
KRAKEN_WS = "wss://ws.kraken.com/v2"
KRAKEN_SYMBOL = "BTC/USD"
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
FEATURE_COLS_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")


def _subscribe_msg(channel: str) -> dict:
    return {"type": "subscribe", "product_ids": [PRODUCT_ID], "channel": channel, "jwt": build_ws_jwt()}


KRAKEN_BOOK_DEPTH = 100  # same fix as kraken_ingestion.py -- was unset, likely
                          # defaulted to a shallow depth that silently capped
                          # imbalance_20/imbalance_50


def _kraken_subscribe_msg(channel: str) -> dict:
    params = {"channel": channel, "symbol": [KRAKEN_SYMBOL]}
    if channel == "book":
        params["depth"] = KRAKEN_BOOK_DEPTH
    return {"method": "subscribe", "params": params}


CRYPTOCOM_WS = "wss://stream.crypto.com/exchange/v1/market"
CRYPTOCOM_INSTRUMENT = "BTC_USD"
CRYPTOCOM_BOOK_DEPTH = 50


def _cryptocom_subscribe_msg() -> dict:
    return {
        "id": 1,
        "method": "subscribe",
        "params": {"channels": [f"book.{CRYPTOCOM_INSTRUMENT}.{CRYPTOCOM_BOOK_DEPTH}", f"trade.{CRYPTOCOM_INSTRUMENT}"]},
    }


def _cryptocom_try_parse_trade(entry: dict) -> tuple[float, float] | None:
    try:
        price = entry.get("p", entry.get("price"))
        qty = entry.get("q", entry.get("quantity", entry.get("qty")))
        if price is None or qty is None:
            return None
        return float(price), float(qty)
    except (TypeError, ValueError):
        return None


app = FastAPI(title="BTC 15-min Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your dashboard's origin once deployed
    allow_methods=["GET"],
)

STATE = {
    "buf": RollingBuffer(max_seconds=900),
    "window_id": None,
    "strike_price": None,
    "directional_model": None,
    "flip_model": None,
    "feature_cols": [],
}

KRAKEN_STATE = {
    "buf": RollingBuffer(max_seconds=900),
    "window_id": None,
    "strike_price": None,
}

CRYPTOCOM_STATE = {
    "buf": RollingBuffer(max_seconds=900),
    "window_id": None,
    "strike_price": None,
}

KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
KALSHI_SERIES_TICKER = "KXBTC15M"
KALSHI_POLL_INTERVAL_SEC = 3.0

KALSHI_STATE = {
    "strike": None,   # Kalshi's own "floor_strike" -- the actual settlement
                       # threshold/target, labeled "Target" in their UI. This
                       # is the real number that determines up/down, not our
                       # own locally-observed window-open price.
    "ticker": None,
    "yes_bid_cents": None,
    "yes_ask_cents": None,
}


def load_models():
    feature_cols = []
    if os.path.exists(FEATURE_COLS_PATH):
        with open(FEATURE_COLS_PATH) as f:
            feature_cols = json.load(f).get("feature_cols", [])

    directional_model = None
    flip_model = None
    dpath = os.path.join(ARTIFACT_DIR, "directional_model.json")
    fpath = os.path.join(ARTIFACT_DIR, "flip_model.json")
    if os.path.exists(dpath):
        directional_model = xgb.XGBClassifier()
        directional_model.load_model(dpath)
    if os.path.exists(fpath):
        flip_model = xgb.XGBClassifier()
        flip_model.load_model(fpath)

    STATE["directional_model"] = directional_model
    STATE["flip_model"] = flip_model
    STATE["feature_cols"] = feature_cols
    log.info(f"Models loaded: directional={directional_model is not None} flip={flip_model is not None}")


async def feed_loop():
    """Background task: maintains the live rolling buffer + order book from Coinbase."""
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(
                ADVANCED_TRADE_WS, ping_interval=20, ping_timeout=20,
                max_size=None,  # level2 snapshot can exceed the 1MB default limit
            ) as ws:
                backoff = 1.0
                await ws.send(json.dumps(_subscribe_msg("market_trades")))
                await ws.send(json.dumps(_subscribe_msg("level2")))
                book = OrderBook()
                last_trade_price = None

                async for raw in ws:
                    msg = json.loads(raw)
                    channel = msg.get("channel")
                    now = datetime.now(timezone.utc)
                    now_ts = now.timestamp()
                    trade_volume_this_message = 0.0

                    if channel == "l2_data":
                        for event in msg.get("events", []):
                            book.apply_event(event)

                    elif channel == "market_trades":
                        for event in msg.get("events", []):
                            if event.get("type") != "update":
                                continue
                            for trade in event.get("trades", []):
                                try:
                                    price = float(trade["price"])
                                    volume = float(trade["size"])
                                except (KeyError, TypeError, ValueError):
                                    continue
                                last_trade_price = price
                                trade_volume_this_message += volume

                    # Fires on EVERY message (book or trade), not just trade
                    # arrival -- same fix as run_ingestion.py's standalone
                    # service. Coinbase's level2 book updates independently
                    # of trades; gating on trade arrival meant book-only
                    # movement never reached the live /live response.
                    # trade_volume_this_message is 0.0 for book-only
                    # messages (correct -- no trade happened at that exact
                    # instant) and the real trade size when one did.
                    if last_trade_price is not None:
                        tick = Tick(
                            ts=now_ts, price=last_trade_price, volume=trade_volume_this_message,
                            best_bid=book.best_bid(), best_ask=book.best_ask(),
                            bid_depth_top10=book.top10_bid_depth(),
                            ask_depth_top10=book.top10_ask_depth(),
                            bid_depth_5=book.bid_depth(5), ask_depth_5=book.ask_depth(5),
                            bid_depth_20=book.bid_depth(20), ask_depth_20=book.ask_depth(20),
                            bid_depth_50=book.bid_depth(50), ask_depth_50=book.ask_depth(50),
                        )
                        STATE["buf"].add(tick)

                        wid = window_id_for(now)
                        if STATE["window_id"] != wid:
                            STATE["window_id"] = wid
                            STATE["strike_price"] = last_trade_price
                            log.info(f"[serving] window rolled: {wid} strike={last_trade_price}")

        except Exception as e:
            log.error(f"[serving] feed error: {e}, retrying in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


async def kraken_feed_loop():
    """Background task: same pattern as feed_loop(), but for Kraken --
    fully independent buffer/window/strike tracking, so an issue on one
    exchange's connection can never affect the other."""
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(KRAKEN_WS, ping_interval=20, ping_timeout=20, max_size=None) as ws:
                backoff = 1.0
                await ws.send(json.dumps(_kraken_subscribe_msg("trade")))
                await ws.send(json.dumps(_kraken_subscribe_msg("book")))
                book = KrakenOrderBook()
                last_trade_price = None

                async for raw in ws:
                    msg = json.loads(raw)
                    channel = msg.get("channel")
                    now = datetime.now(timezone.utc)
                    now_ts = now.timestamp()
                    trade_volume_this_message = 0.0

                    if channel == "book":
                        msg_type = msg.get("type")
                        for entry in msg.get("data", []):
                            book.apply_message(msg_type, entry)

                    elif channel == "trade":
                        for entry in msg.get("data", []):
                            try:
                                price = float(entry["price"])
                                volume = float(entry["qty"])
                            except (KeyError, TypeError, ValueError):
                                continue
                            last_trade_price = price
                            trade_volume_this_message += volume

                    # Fires on EVERY message (book or trade) -- same fix as
                    # kraken_ingestion.py's standalone service.
                    if last_trade_price is not None:
                        tick = Tick(
                            ts=now_ts, price=last_trade_price, volume=trade_volume_this_message,
                            best_bid=book.best_bid(), best_ask=book.best_ask(),
                            bid_depth_top10=book.bid_depth(10), ask_depth_top10=book.ask_depth(10),
                            bid_depth_5=book.bid_depth(5), ask_depth_5=book.ask_depth(5),
                            bid_depth_20=book.bid_depth(20), ask_depth_20=book.ask_depth(20),
                            bid_depth_50=book.bid_depth(50), ask_depth_50=book.ask_depth(50),
                        )
                        KRAKEN_STATE["buf"].add(tick)

                        wid = window_id_for(now)
                        if KRAKEN_STATE["window_id"] != wid:
                            KRAKEN_STATE["window_id"] = wid
                            KRAKEN_STATE["strike_price"] = last_trade_price
                            log.info(f"[serving/kraken] window rolled: {wid} strike={last_trade_price}")

        except Exception as e:
            log.error(f"[serving/kraken] feed error: {e}, retrying in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


async def cryptocom_feed_loop():
    """Same pattern as feed_loop()/kraken_feed_loop() -- fully independent
    state, so this being unverified/experimental can't affect Coinbase or
    Kraken. See crypto_ingestion.py's module docstring for the honest
    caveat about this integration's confidence level."""
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(CRYPTOCOM_WS, ping_interval=20, ping_timeout=20, max_size=None) as ws:
                backoff = 1.0
                await asyncio.sleep(1.0)
                await ws.send(json.dumps(_cryptocom_subscribe_msg()))
                book = CryptoComOrderBook()
                last_trade_price = None

                async for raw in ws:
                    msg = json.loads(raw)

                    if msg.get("method") == "public/heartbeat":
                        await ws.send(json.dumps({"id": msg.get("id"), "method": "public/respond-heartbeat"}))
                        continue

                    result = msg.get("result", {})
                    channel = result.get("channel") or msg.get("channel")
                    now = datetime.now(timezone.utc)
                    now_ts = now.timestamp()
                    trade_volume_this_message = 0.0

                    if channel == "book":
                        for entry in result.get("data", []):
                            book.apply_message(entry, is_snapshot=True)

                    elif channel == "trade":
                        for entry in result.get("data", []):
                            parsed = _cryptocom_try_parse_trade(entry)
                            if parsed is None:
                                continue
                            price, volume = parsed
                            last_trade_price = price
                            trade_volume_this_message += volume

                    # Fires on EVERY message (book or trade), not just trade
                    # arrival -- same fix as crypto_ingestion.py. The book
                    # refreshes on its own ~500ms cadence independent of
                    # trades; gating on trade arrival meant book-only
                    # movement never reached the live /live response either.
                    if last_trade_price is not None:
                        tick = Tick(
                            ts=now_ts, price=last_trade_price, volume=trade_volume_this_message,
                            best_bid=book.best_bid(), best_ask=book.best_ask(),
                            bid_depth_top10=book.bid_depth(10), ask_depth_top10=book.ask_depth(10),
                            bid_depth_5=book.bid_depth(5), ask_depth_5=book.ask_depth(5),
                            bid_depth_20=book.bid_depth(20), ask_depth_20=book.ask_depth(20),
                            bid_depth_50=book.bid_depth(50), ask_depth_50=book.ask_depth(50),
                        )
                        CRYPTOCOM_STATE["buf"].add(tick)

                        wid = window_id_for(now)
                        if CRYPTOCOM_STATE["window_id"] != wid:
                            CRYPTOCOM_STATE["window_id"] = wid
                            CRYPTOCOM_STATE["strike_price"] = last_trade_price
                            log.info(f"[serving/cryptocom] window rolled: {wid} strike={last_trade_price}")

        except Exception as e:
            log.error(f"[serving/cryptocom] feed error: {e}, retrying in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def _fetch_kalshi_market() -> dict | None:
    """Synchronous, blocking call -- run via loop.run_in_executor so it
    never freezes the shared asyncio event loop that Coinbase, Kraken, and
    Crypto.com's WebSocket feeds also depend on."""
    resp = requests.get(
        f"{KALSHI_BASE_URL}/markets",
        params={"series_ticker": KALSHI_SERIES_TICKER, "status": "open", "limit": 5},
        timeout=5,
    )
    resp.raise_for_status()
    markets = resp.json().get("markets", [])
    return markets[0] if markets else None


async def kalshi_poll_loop():
    """Polls Kalshi's public REST API for the currently open KXBTC15M
    market's floor_strike (the real settlement threshold/"Target" in their
    UI) -- this is a much lighter-weight poll than the full kalshi_poller.py
    service, since the dashboard only needs the strike + current bid/ask,
    not a full historical price log."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            m = await loop.run_in_executor(None, _fetch_kalshi_market)
            if m:
                strike = m.get("floor_strike")
                yes_bid = m.get("yes_bid_dollars")
                yes_ask = m.get("yes_ask_dollars")

                if strike is not None:
                    KALSHI_STATE["strike"] = float(strike)
                KALSHI_STATE["ticker"] = m.get("ticker")
                if yes_bid is not None:
                    KALSHI_STATE["yes_bid_cents"] = float(yes_bid) * 100.0
                if yes_ask is not None:
                    KALSHI_STATE["yes_ask_cents"] = float(yes_ask) * 100.0
        except Exception as e:
            log.warning(f"[serving/kalshi] poll error: {e}")

        await asyncio.sleep(KALSHI_POLL_INTERVAL_SEC)


@app.on_event("startup")
async def startup():
    load_models()
    asyncio.create_task(feed_loop())
    asyncio.create_task(kraken_feed_loop())
    asyncio.create_task(kalshi_poll_loop())
    asyncio.create_task(cryptocom_feed_loop())


@app.get("/health")
async def health():
    return {"ok": True, "window_id": STATE["window_id"], "ticks_buffered": len(STATE["buf"].ticks)}


@app.get("/live")
async def live():
    buf = STATE["buf"]
    if not buf.ticks or STATE["strike_price"] is None:
        return {"status": "warming_up"}

    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()
    feats = compute_feature_snapshot(buf, STATE["strike_price"], now_ts)
    secs_remaining = seconds_remaining(now, STATE["window_id"])
    feats["seconds_remaining"] = secs_remaining

    result = {
        "window_id": STATE["window_id"],
        "strike_price": STATE["strike_price"],
        "current_price": feats.get("price"),
        "seconds_remaining": secs_remaining,
        "features": feats,
        "directional": None,
        "flip": None,
        "kraken": None,
        "kalshi_strike": KALSHI_STATE["strike"],
        "kalshi_ticker": KALSHI_STATE["ticker"],
        "kalshi_yes_bid_cents": KALSHI_STATE["yes_bid_cents"],
        "kalshi_yes_ask_cents": KALSHI_STATE["yes_ask_cents"],
        "cryptocom": None,
    }

    kraken_buf = KRAKEN_STATE["buf"]
    if kraken_buf.ticks and KRAKEN_STATE["strike_price"] is not None:
        kraken_feats = compute_feature_snapshot(kraken_buf, KRAKEN_STATE["strike_price"], now_ts)
        result["kraken"] = {
            "window_id": KRAKEN_STATE["window_id"],
            "strike_price": KRAKEN_STATE["strike_price"],
            "current_price": kraken_feats.get("price"),
            "features": kraken_feats,
        }

    cryptocom_buf = CRYPTOCOM_STATE["buf"]
    if cryptocom_buf.ticks and CRYPTOCOM_STATE["strike_price"] is not None:
        cryptocom_feats = compute_feature_snapshot(cryptocom_buf, CRYPTOCOM_STATE["strike_price"], now_ts)
        result["cryptocom"] = {
            "window_id": CRYPTOCOM_STATE["window_id"],
            "strike_price": CRYPTOCOM_STATE["strike_price"],
            "current_price": cryptocom_feats.get("price"),
            "features": cryptocom_feats,
        }

    cols = STATE["feature_cols"]
    if STATE["directional_model"] is not None and cols:
        row = [[feats.get(c.replace("feat_", ""), None) for c in cols]]
        try:
            import numpy as np
            X = np.array(row, dtype=float)
            if not np.isnan(X).any():
                p_up = float(STATE["directional_model"].predict_proba(X)[0, 1])
                result["directional"] = {"p_up": p_up, "p_down": 1 - p_up}
        except Exception as e:
            result["directional"] = {"error": str(e)}

    if STATE["flip_model"] is not None and cols:
        try:
            import numpy as np
            row = [[feats.get(c.replace("feat_", ""), None) for c in cols]]
            X = np.array(row, dtype=float)
            if not np.isnan(X).any():
                p_reversal = float(STATE["flip_model"].predict_proba(X)[0, 1])
                result["flip"] = {"p_flip": p_reversal}
        except Exception as e:
            result["flip"] = {"error": str(e)}

    return result
