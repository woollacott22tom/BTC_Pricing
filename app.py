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
  GET /live       -> current window's live reversal score
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
from compute import (
    Tick, RollingBuffer, compute_feature_snapshot, compute_mean_surge_indicator,
    build_live_feature_row, RollingSeries, compute_price_diff_deviation_features,
    compute_kalshi_momentum_features, book_imbalance,
    compute_block_summary, segment_chunks, build_lookback_features,
)
from dynamo_client import list_closed_windows
import boto3
from boto3.dynamodb.conditions import Key
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
    "strike_cross_model": None,
    "feature_cols": [],
}

REGION = os.environ.get("AWS_REGION", "us-east-1")
ORDER_BOOK_FIX_CUTOFF_TS = 1787118936.0  # matches train.py -- 2026-08-19T05:55:36+00:00 UTC
CONTINUATION_HISTORY_SIZE = 30  # recent CLOSED windows kept for chunk/lookback context
CONTINUATION_REFRESH_INTERVAL_SEC = 300.0  # window outcomes only change once per 15 min,
                                             # no need to refetch on every /live poll

CONTINUATION_STATE = {
    "model": None,
    "feature_cols": None,
    "window_history": [],  # list of (block_summary_dict, direction) tuples, chronological
    "last_refresh_ts": 0.0,
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
    # Rolling history of Kalshi's own mid price -- needed to compute
    # kalshi_momentum_15s/60s live, the same way train.py computes it from
    # stored ticks. Bounded to 90s: comfortably covers the 60s lookback
    # plus margin, without keeping unbounded history in memory.
    "price_history": RollingSeries(max_seconds=90.0),
}

# Rolling cross-exchange price-gap history -- needed to compute the
# price_diff_*_deviation_* features live, the same way
# add_price_differential_features() does from stored ticks. Bounded to
# 150s: comfortably covers the largest (120s) rolling baseline plus
# margin. Persists across /live calls (module-level state) so the rolling
# history actually accumulates over time rather than resetting every poll.
PRICE_DIFF_STATE = {
    "cc_cb": RollingSeries(max_seconds=150.0),
    "kr_cb": RollingSeries(max_seconds=150.0),
}


def get_generic_window_ticks(table_name: str, window_id: str) -> list:
    """Same query pattern as train.py's function of the same name --
    duplicated here rather than imported since it's a tiny, self-contained
    boto3 query with no shared logic worth factoring out across the
    training/serving process boundary."""
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(table_name)
    items = []
    resp = table.query(KeyConditionExpression=Key("window_id").eq(window_id))
    items.extend(resp["Items"])
    while "LastEvaluatedKey" in resp:
        resp = table.query(
            KeyConditionExpression=Key("window_id").eq(window_id),
            ExclusiveStartKey=resp["LastEvaluatedKey"],
        )
        items.extend(resp["Items"])
    return sorted(items, key=lambda x: float(x["timestamp"]))


def refresh_continuation_history():
    """Fetches the last CONTINUATION_HISTORY_SIZE CLOSED windows, computes
    each one's block summary + direction, and stores the result for live
    Live/Future scoring. Synchronous/blocking -- always called via
    run_in_executor so it never freezes the shared event loop the
    WebSocket feeds depend on."""
    try:
        windows = list_closed_windows(region_name=REGION)
        windows = [w for w in windows if w.get("source") != "backfill"]
        windows = [
            w for w in windows
            if datetime.fromisoformat(w["window_id"]).timestamp() + 900.0 > ORDER_BOOK_FIX_CUTOFF_TS
        ]
        windows = sorted(windows, key=lambda w: w["window_id"])[-CONTINUATION_HISTORY_SIZE:]

        history = []
        for w in windows:
            ticks = get_generic_window_ticks("btc_ticks", w["window_id"])
            if not ticks:
                continue
            ticks_sorted = sorted(ticks, key=lambda t: float(t["timestamp"]))
            strike_price = float(ticks_sorted[0]["price"])
            summary = compute_block_summary(w["window_id"], ticks_sorted, strike_price)
            if summary is None:
                continue
            true_outcome = w.get("kalshi_true_outcome")
            outcome = true_outcome or w.get("outcome")
            history.append((summary, 1 if outcome == "up" else -1))

        CONTINUATION_STATE["window_history"] = history
        CONTINUATION_STATE["last_refresh_ts"] = time.time()
        log.info(f"[continuation] refreshed window history: {len(history)} closed windows summarized")
    except Exception as e:
        log.warning(f"[continuation] history refresh failed: {e}")


async def continuation_history_refresh_loop():
    loop = asyncio.get_event_loop()
    while True:
        await loop.run_in_executor(None, refresh_continuation_history)
        await asyncio.sleep(CONTINUATION_REFRESH_INTERVAL_SEC)


def compute_live_block_summary(buf, window_id: str, window_start_ts: float, strike_price: float):
    """Builds the SAME dict shape compute_block_summary expects, from the
    LIVE, in-progress tick buffer -- filtered to only this window's own
    ticks so far. Reuses book_imbalance() (the exact same function used
    at ingestion/training time) per tick, rather than reimplementing the
    formula, to avoid any risk of live/train skew."""
    ticks_in_window = [t for t in buf.ticks if t.ts >= window_start_ts]
    if len(ticks_in_window) < 10:
        return None
    tick_dicts = [
        {"timestamp": t.ts, "price": t.price, "volume": t.volume or 0.0, "book_imbalance": book_imbalance(t)}
        for t in ticks_in_window
    ]
    return compute_block_summary(window_id, tick_dicts, strike_price)


def score_continuation_live():
    """Returns (p_live, p_future, live_trend_direction).

    Live: does the CURRENTLY OPEN window continue the trend established
    by the last CLOSED window -- built entirely from settled, ground-
    truth history. Stable; only changes once every 15 minutes, right
    when a window closes. live_trend_direction is that established
    trend's OWN direction (chunk_direction of the last closed window) --
    exposed so the dashboard can color Live green/red based on which way
    the trend it's actually predicting FROM is pointing, not the
    probability value itself.

    Future: does the window AFTER NEXT continue what's forming in the
    CURRENTLY OPEN window's own live, partial, evolving data -- the
    currently-open window is appended as a PROVISIONAL entry, with its
    direction INFERRED from current price vs. its own strike (an
    approximation, since it hasn't actually settled yet). Updates
    continuously as the window's own data accumulates."""
    model = CONTINUATION_STATE["model"]
    feature_cols = CONTINUATION_STATE["feature_cols"]
    history = CONTINUATION_STATE["window_history"]
    if model is None or not feature_cols or len(history) < 3:
        return None, None, None

    import numpy as np
    block_rows = [h[0] for h in history]
    directions = [h[1] for h in history]

    p_live = None
    live_trend_direction = None
    try:
        chunk_ids = segment_chunks(directions)
        feature_rows = build_lookback_features(block_rows, directions, chunk_ids)
        if feature_rows:
            last_row = feature_rows[-1]
            X = np.array([[last_row.get(c, np.nan) for c in feature_cols]], dtype=float)
            p_live = float(model.predict_proba(X)[0, 1])
            live_trend_direction = "up" if last_row["chunk_direction"] > 0 else "down"
    except Exception as e:
        log.warning(f"[continuation] live scoring failed: {e}")

    p_future = None
    try:
        if STATE["window_id"] is not None and STATE["strike_price"] is not None:
            window_start_ts = datetime.fromisoformat(STATE["window_id"]).timestamp()
            live_summary = compute_live_block_summary(
                STATE["buf"], STATE["window_id"], window_start_ts, STATE["strike_price"],
            )
            if live_summary is not None:
                inferred_direction = 1 if live_summary["vwap"] >= STATE["strike_price"] else -1
                provisional_rows = block_rows + [live_summary]
                provisional_directions = directions + [inferred_direction]
                provisional_chunk_ids = segment_chunks(provisional_directions)
                provisional_features = build_lookback_features(
                    provisional_rows, provisional_directions, provisional_chunk_ids,
                )
                if provisional_features:
                    future_row = provisional_features[-1]
                    X2 = np.array([[future_row.get(c, np.nan) for c in feature_cols]], dtype=float)
                    p_future = float(model.predict_proba(X2)[0, 1])
    except Exception as e:
        log.warning(f"[continuation] future scoring failed: {e}")

    return p_live, p_future, live_trend_direction


def load_models():
    """NOTE: directional model intentionally not loaded/served anymore --
    settlement prediction was scrapped in favor of strike-crossing and
    block-level continuation. reversal (flip_model) is also no longer
    loaded/served -- confirmed at chance-level AUC (~0.50) across two
    separate full training runs on real data, meaning a probability from
    it would look like a real signal without actually being one."""
    feature_cols = []
    if os.path.exists(FEATURE_COLS_PATH):
        with open(FEATURE_COLS_PATH) as f:
            feature_cols = json.load(f).get("feature_cols", [])

    strike_cross_model = None
    scpath = os.path.join(ARTIFACT_DIR, "strike_cross_model.json")
    if os.path.exists(scpath):
        strike_cross_model = xgb.XGBClassifier()
        strike_cross_model.load_model(scpath)

    continuation_model = None
    continuation_feature_cols = None
    cpath = os.path.join(ARTIFACT_DIR, "continuation_model.json")
    if os.path.exists(cpath):
        continuation_model = xgb.XGBClassifier()
        continuation_model.load_model(cpath)
        if os.path.exists(FEATURE_COLS_PATH):
            with open(FEATURE_COLS_PATH) as f:
                continuation_feature_cols = json.load(f).get("continuation_feature_cols")

    STATE["strike_cross_model"] = strike_cross_model
    STATE["feature_cols"] = feature_cols
    CONTINUATION_STATE["model"] = continuation_model
    CONTINUATION_STATE["feature_cols"] = continuation_feature_cols
    log.info(f"Models loaded: strike_cross={strike_cross_model is not None} "
             f"continuation={continuation_model is not None}")


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
                pending_volume = 0.0
                last_logged = 0.0

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
                                continue
                            for trade in event.get("trades", []):
                                try:
                                    price = float(trade["price"])
                                    volume = float(trade["size"])
                                except (KeyError, TypeError, ValueError):
                                    continue
                                last_trade_price = price
                                pending_volume += volume

                    # Fires on a ~1s heartbeat, not every raw message.
                    # PREVIOUS version fired on every message (any channel),
                    # unbounded -- Coinbase's level2 feed can send far more
                    # than 1 message/sec, and since RollingBuffer is bounded
                    # by TIME not tick COUNT, that let the buffer grow much
                    # larger than intended, making each /live request's
                    # feature computation progressively more expensive and
                    # plausibly starving the event loop enough to cause the
                    # instability observed (3 unplanned restarts, rising
                    # CPU load). This still captures book-only movement
                    # (the original point of the fix) since the heartbeat
                    # fires regardless of which message type triggered it --
                    # it just doesn't fire on literally every one anymore.
                    if last_trade_price is not None and now_ts - last_logged >= 1.0:
                        tick = Tick(
                            ts=now_ts, price=last_trade_price, volume=pending_volume,
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

                        last_logged = now_ts
                        pending_volume = 0.0

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
                pending_volume = 0.0
                last_logged = 0.0

                async for raw in ws:
                    msg = json.loads(raw)
                    channel = msg.get("channel")
                    now = datetime.now(timezone.utc)
                    now_ts = now.timestamp()

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
                            pending_volume += volume

                    # Fires on a ~1s heartbeat, not every raw message --
                    # same fix and same reasoning as feed_loop() above.
                    if last_trade_price is not None and now_ts - last_logged >= 1.0:
                        tick = Tick(
                            ts=now_ts, price=last_trade_price, volume=pending_volume,
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

                        last_logged = now_ts
                        pending_volume = 0.0

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
                pending_volume = 0.0
                last_logged = 0.0

                async for raw in ws:
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
                            book.apply_message(entry, is_snapshot=True)

                    elif channel == "trade":
                        for entry in result.get("data", []):
                            parsed = _cryptocom_try_parse_trade(entry)
                            if parsed is None:
                                continue
                            price, volume = parsed
                            last_trade_price = price
                            pending_volume += volume

                    # Fires on a ~1s heartbeat, not every raw message --
                    # same fix and same reasoning as feed_loop() above.
                    if last_trade_price is not None and now_ts - last_logged >= 1.0:
                        tick = Tick(
                            ts=now_ts, price=last_trade_price, volume=pending_volume,
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

                        last_logged = now_ts
                        pending_volume = 0.0

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

                # Feed the rolling price history used for live momentum --
                # mid price, same convention as kalshi_poller.py's stored
                # yes_mid_cents (average of bid and ask).
                if yes_bid is not None and yes_ask is not None:
                    mid_cents = (float(yes_bid) + float(yes_ask)) * 100.0 / 2.0
                    KALSHI_STATE["price_history"].add(time.time(), mid_cents)
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
    asyncio.create_task(continuation_history_refresh_loop())


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
        "strike_cross": None,
        "continuation": None,
        "mean_surge": None,
        "kraken": None,
        "kalshi_strike": KALSHI_STATE["strike"],
        "kalshi_ticker": KALSHI_STATE["ticker"],
        "kalshi_yes_bid_cents": KALSHI_STATE["yes_bid_cents"],
        "kalshi_yes_ask_cents": KALSHI_STATE["yes_ask_cents"],
        "cryptocom": None,
    }

    # Mean/surge signal -- validated in mean_surge_outcome_analysis.py.
    # See compute_mean_surge_indicator's docstring for the exact backtest
    # numbers behind the strong/weak classification.
    window_start_ts = datetime.fromisoformat(STATE["window_id"]).timestamp()
    mean_surge = compute_mean_surge_indicator(buf, STATE["strike_price"], window_start_ts, now_ts)
    if mean_surge is not None:
        result["mean_surge"] = mean_surge

    kraken_feats = None
    kraken_buf = KRAKEN_STATE["buf"]
    if kraken_buf.ticks and KRAKEN_STATE["strike_price"] is not None:
        kraken_feats = compute_feature_snapshot(kraken_buf, KRAKEN_STATE["strike_price"], now_ts)
        result["kraken"] = {
            "window_id": KRAKEN_STATE["window_id"],
            "strike_price": KRAKEN_STATE["strike_price"],
            "current_price": kraken_feats.get("price"),
            "features": kraken_feats,
        }

    cryptocom_feats = None
    cryptocom_buf = CRYPTOCOM_STATE["buf"]
    if cryptocom_buf.ticks and CRYPTOCOM_STATE["strike_price"] is not None:
        cryptocom_feats = compute_feature_snapshot(cryptocom_buf, CRYPTOCOM_STATE["strike_price"], now_ts)
        result["cryptocom"] = {
            "window_id": CRYPTOCOM_STATE["window_id"],
            "strike_price": CRYPTOCOM_STATE["strike_price"],
            "current_price": cryptocom_feats.get("price"),
            "features": cryptocom_feats,
        }

    # Row-building done ONCE, reused for both models below. Important:
    # compute_price_diff_deviation_features() has a side effect -- it
    # appends the current gap to persistent rolling-history state on every
    # call. Calling it separately per model would double-count the same
    # tick into that history, corrupting the rolling mean. build_live_feature_row()
    # builds the prefixed + consensus column set; price-diff deviation and
    # Kalshi momentum need persistent rolling-history state (not just a
    # snapshot), computed separately and merged in. Together these cover
    # every column train.py trained on, verified against real pandas
    # ground truth for the rolling/momentum boundary conventions
    # specifically (pandas uses TWO DIFFERENT window conventions across
    # train.py -- .rolling() is exclusive-start, the momentum boolean
    # filter is inclusive-both-ends -- confirmed empirically before
    # shipping, not assumed). Any column still genuinely unavailable
    # (e.g. too little history yet after a fresh restart) is left as NaN;
    # XGBoost handles missing values natively rather than refusing to
    # score at all, which is what the original all-or-nothing gate did.
    cols = STATE["feature_cols"]
    if cols and STATE["strike_cross_model"] is not None:
        try:
            import numpy as np
            row_dict = build_live_feature_row(
                feats, kraken_feats, cryptocom_feats,
                hour_of_day_utc=now.hour, day_of_week=now.weekday(),
            )

            price_diff_row = compute_price_diff_deviation_features(
                PRICE_DIFF_STATE,
                cb_price=feats.get("price"),
                kr_price=kraken_feats.get("price") if kraken_feats else None,
                cc_price=cryptocom_feats.get("price") if cryptocom_feats else None,
                now_ts=now_ts,
            )
            row_dict.update(price_diff_row)

            kalshi_momentum_row = compute_kalshi_momentum_features(
                KALSHI_STATE["price_history"], now_ts,
            )
            row_dict.update(kalshi_momentum_row)

            row = [[row_dict.get(c, np.nan) for c in cols]]
            X = np.array(row, dtype=float)

            try:
                p_cross = float(STATE["strike_cross_model"].predict_proba(X)[0, 1])
                result["strike_cross"] = {"p_cross": p_cross}
            except Exception as e:
                result["strike_cross"] = {"error": str(e)}
        except Exception as e:
            log.warning(f"[serving] feature-row build failed: {e}")

    if CONTINUATION_STATE["model"] is not None:
        p_live, p_future, live_trend_direction = score_continuation_live()
        result["continuation"] = {
            "p_live": p_live, "p_future": p_future, "live_trend_direction": live_trend_direction,
        }

    return result
