"""
Live scoring API. Runs on the same EC2 box as ingestion (or a second
process) -- maintains its own rolling buffer from the same public Binance
feed, loads trained model artifacts, and exposes a small JSON endpoint your
GitHub-hosted dashboard can poll.

    uvicorn serving.app:app --host 0.0.0.0 --port 8000

CORS is left open (`*`) since the dashboard is a static site on a different
origin (GitHub Pages / S3). Tighten `allow_origins` to your actual dashboard
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
import sys
import time
from datetime import datetime, timezone

import websockets
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from window_utils import window_id_for, seconds_remaining
from compute import Tick, RollingBuffer, compute_feature_snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serving")

BINANCE_WS = "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/btcusdt@depth20@100ms"
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
FEATURE_COLS_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")

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
    "last_depth": None,
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
    """Background task: maintains the live rolling buffer from Binance."""
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(BINANCE_WS, ping_interval=20, ping_timeout=20) as ws:
                backoff = 1.0
                async for raw in ws:
                    msg = json.loads(raw)
                    stream = msg.get("stream", "")
                    data = msg.get("data", {})
                    now = datetime.now(timezone.utc)
                    now_ts = now.timestamp()

                    if stream.endswith("@trade"):
                        price = float(data["p"])
                        volume = float(data["q"])
                        depth = STATE["last_depth"] or {}
                        tick = Tick(
                            ts=now_ts, price=price, volume=volume,
                            best_bid=depth.get("best_bid"), best_ask=depth.get("best_ask"),
                            bid_depth_top10=depth.get("bid"), ask_depth_top10=depth.get("ask"),
                        )
                        STATE["buf"].add(tick)

                        wid = window_id_for(now)
                        if STATE["window_id"] != wid:
                            STATE["window_id"] = wid
                            STATE["strike_price"] = price
                            log.info(f"[serving] window rolled: {wid} strike={price}")

                    elif stream.endswith("@depth20@100ms"):
                        bids = data.get("bids", [])[:10]
                        asks = data.get("asks", [])[:10]
                        if bids and asks:
                            STATE["last_depth"] = {
                                "bid": sum(float(sz) for _, sz in bids),
                                "ask": sum(float(sz) for _, sz in asks),
                                "best_bid": float(bids[0][0]),
                                "best_ask": float(asks[0][0]),
                            }
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
        # NOTE: keys in `feats` don't carry the `feat_` prefix used in training
        # (that prefix is added at DynamoDB write time). This mapping keeps
        # train/serve features aligned -- if you rename anything in
        # features/compute.py, update this mapping too.
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
