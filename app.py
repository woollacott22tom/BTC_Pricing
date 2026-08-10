"""
Live scoring API. Runs on the same EC2 box as ingestion (or a second
process) -- maintains its own rolling buffer + order book from the same
authenticated Coinbase Advanced Trade feed, loads trained model artifacts,
and exposes a small JSON endpoint your GitHub-hosted dashboard can poll.

Requires the same COINBASE_API_KEY_NAME / COINBASE_API_PRIVATE_KEY_PATH
environment variables as run_ingestion.py.

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
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from window_utils import window_id_for, seconds_remaining
from compute import Tick, RollingBuffer, compute_feature_snapshot
from order_book import OrderBook
from jwt_auth import build_ws_jwt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serving")

ADVANCED_TRADE_WS = "wss://advanced-trade-ws.coinbase.com"
PRODUCT_ID = "BTC-USD"
JWT_REFRESH_INTERVAL_SEC = 90
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
FEATURE_COLS_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")


def _subscribe_msg(channel: str) -> dict:
    return {
        "type": "subscribe",
        "product_ids": [PRODUCT_ID],
        "channel": channel,
        "jwt": build_ws_jwt(),
    }


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
            async with websockets.connect(ADVANCED_TRADE_WS, ping_interval=20, ping_timeout=20) as ws:
                backoff = 1.0
                await ws.send(json.dumps(_subscribe_msg("market_trades")))
                await ws.send(json.dumps(_subscribe_msg("level2")))
                book = OrderBook()
                last_jwt_refresh = time.time()

                async for raw in ws:
                    now_check = time.time()
                    if now_check - last_jwt_refresh > JWT_REFRESH_INTERVAL_SEC:
                        await ws.send(json.dumps(_subscribe_msg("market_trades")))
                        await ws.send(json.dumps(_subscribe_msg("level2")))
                        last_jwt_refresh = now_check

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

                                tick = Tick(
                                    ts=now_ts, price=price, volume=volume,
                                    best_bid=book.best_bid(), best_ask=book.best_ask(),
                                    bid_depth_top10=book.top10_bid_depth(),
                                    ask_depth_top10=book.top10_ask_depth(),
                                )
                                STATE["buf"].add(tick)

                                wid = window_id_for(now)
                                if STATE["window_id"] != wid:
                                    STATE["window_id"] = wid
                                    STATE["strike_price"] = price
                                    log.info(f"[serving] window rolled: {wid} strike={price}")

        except Exception as e:
            log.error(f"[serving] feed error: {e}, retrying in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


@app.on_event("startup")
async def startup():
    load_models()
    asyncio.create_task(feed_loop())


@app.get("/health")
def health():
    return {"ok": True, "window_id": STATE["window_id"], "ticks_buffered": len(STATE["buf"].ticks)}


@app.get("/live")
def live():
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

    if STATE["flip_model"] is not None and cols and secs_remaining <= 90:
        try:
            import numpy as np
            row = [[feats.get(c.replace("feat_", ""), None) for c in cols]]
            X = np.array(row, dtype=float)
            if not np.isnan(X).any():
                p_flip = float(STATE["flip_model"].predict_proba(X)[0, 1])
                result["flip"] = {"p_flip": p_flip}
        except Exception as e:
            result["flip"] = {"error": str(e)}

    return result
