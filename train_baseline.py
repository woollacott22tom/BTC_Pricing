"""
Trains a simple baseline directional model using ONLY the OHLCV fields
already present on backfilled window summaries -- no per-tick data needed,
so this doesn't touch btc_ticks at all and runs fast even across 17k+
windows.

This is intentionally a much weaker feature set than the live full-featured
model (no order-flow, no book imbalance, no sub-second momentum) -- it
exists purely as an early sanity check: does directional structure exist
in candle-level features at all, and what's a reasonable floor to expect
before the live full-featured model has enough data to train.

Features derived purely from each window's own OHLCV (all "point-in-time
at window open" safe -- NOT using the window's own close/high/low as
features, since those aren't known until the window ends and would leak
the label):
  - prior_window_return: return of the PREVIOUS window (t-1 close vs open)
  - prior_window_range: (high-low)/open of the previous window
  - rolling_3_window_momentum: return over the last 3 windows combined
  - hour_of_day, day_of_week: simple seasonality features

Usage:
    python3 train_baseline.py
"""
from __future__ import annotations
import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
import xgboost as xgb
import boto3

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dynamo_client import list_closed_windows

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_baseline")

REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("MODEL_BUCKET", "btc-kalshi-model-artifacts-tom-8291")
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def load_backfill_dataframe() -> pd.DataFrame:
    windows = list_closed_windows(region_name=REGION)
    backfill = [w for w in windows if w.get("source") == "backfill"]
    log.info(f"Found {len(backfill)} backfilled windows (out of {len(windows)} total)")

    rows = []
    for w in backfill:
        try:
            rows.append({
                "window_id": w["window_id"],
                "open": float(w["open_price"]),
                "close": float(w["settlement_price"]),
                "high": float(w.get("high", w["open_price"])),
                "low": float(w.get("low", w["open_price"])),
                "volume": float(w.get("volume", 0)),
                "outcome_up": 1 if w.get("outcome") == "up" else 0,
            })
        except (KeyError, TypeError, ValueError):
            continue

    df = pd.DataFrame(rows).sort_values("window_id").reset_index(drop=True)
    df["window_dt"] = pd.to_datetime(df["window_id"])
    log.info(f"Built dataframe: {len(df)} rows")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["window_return"] = (df["close"] - df["open"]) / df["open"]
    df["window_range"] = (df["high"] - df["low"]) / df["open"]

    # shift(1) ensures we only use the PRIOR window's data -- using the
    # current window's own close/high/low as a feature would leak the label
    df["prior_window_return"] = df["window_return"].shift(1)
    df["prior_window_range"] = df["window_range"].shift(1)
    df["prior_volume"] = df["volume"].shift(1)
    df["rolling_3_window_momentum"] = df["window_return"].shift(1).rolling(3).sum()
    df["rolling_6_window_momentum"] = df["window_return"].shift(1).rolling(6).sum()

    df["hour_of_day"] = df["window_dt"].dt.hour
    df["day_of_week"] = df["window_dt"].dt.dayofweek

    return df


FEATURE_COLS = [
    "prior_window_return", "prior_window_range", "prior_volume",
    "rolling_3_window_momentum", "rolling_6_window_momentum",
    "hour_of_day", "day_of_week",
]


def walk_forward_splits(df: pd.DataFrame, n_splits: int = 6):
    """Time-ordered splits -- train on earlier windows, test on the next
    chronological chunk. Never shuffle, since adjacent windows are
    correlated (volatility clusters) and a random split would leak."""
    n = len(df)
    fold_size = max(1, n // (n_splits + 1))
    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        test_end = min((i + 1) * fold_size, n)
        if test_end <= train_end:
            continue
        yield df.iloc[:train_end], df.iloc[train_end:test_end]


def train_and_evaluate(df: pd.DataFrame):
    df = df.dropna(subset=FEATURE_COLS + ["outcome_up"])
    log.info(f"Rows after dropping NaN (from rolling/shift warmup): {len(df)}")

    aucs, briers = [], []
    for train_df, test_df in walk_forward_splits(df):
        if train_df.empty or test_df.empty or test_df["outcome_up"].nunique() < 2:
            continue
        model = xgb.XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        )
        model.fit(train_df[FEATURE_COLS], train_df["outcome_up"])
        preds = model.predict_proba(test_df[FEATURE_COLS])[:, 1]
        aucs.append(roc_auc_score(test_df["outcome_up"], preds))
        briers.append(brier_score_loss(test_df["outcome_up"], preds))

    log.info(f"Walk-forward AUC: {np.mean(aucs):.4f} (0.5 = no better than a coin flip)")
    log.info(f"Walk-forward Brier: {np.mean(briers):.4f} (lower is better calibrated; "
              f"0.25 is what always-predicting-50% gives you)")

    baseline_up_rate = df["outcome_up"].mean()
    log.info(f"Base rate of 'up' outcomes in this data: {baseline_up_rate:.4f} "
              f"(compare against 0.5 -- a big deviation here suggests a labeling or data issue, not a real edge)")

    final_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    )
    final_model.fit(df[FEATURE_COLS], df["outcome_up"])

    return final_model, {
        "auc": float(np.mean(aucs)) if aucs else None,
        "brier": float(np.mean(briers)) if briers else None,
        "base_rate_up": float(baseline_up_rate),
        "n_rows": len(df),
    }


def save_artifacts(model, metrics: dict):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    model.save_model(os.path.join(ARTIFACT_DIR, "baseline_model.json"))
    with open(os.path.join(ARTIFACT_DIR, "baseline_metadata.json"), "w") as f:
        json.dump({
            "trained_at": datetime.utcnow().isoformat(),
            "feature_cols": FEATURE_COLS,
            "metrics": metrics,
            "note": "Candle-only baseline, trained on backfilled OHLCV data. "
                    "Not comparable in quality to the full live-feature model "
                    "once enough tick-level data has accumulated.",
        }, f, indent=2)

    try:
        s3 = boto3.client("s3", region_name=REGION)
        for fname in ["baseline_model.json", "baseline_metadata.json"]:
            s3.upload_file(os.path.join(ARTIFACT_DIR, fname), S3_BUCKET, f"models/{fname}")
        log.info(f"Uploaded baseline artifacts to s3://{S3_BUCKET}/models/")
    except Exception as e:
        log.warning(f"S3 upload failed (local artifacts still saved): {e}")


def main():
    df = load_backfill_dataframe()
    if len(df) < 500:
        log.warning("Not enough backfilled windows to train a meaningful baseline. Exiting.")
        sys.exit(0)

    df = add_features(df)
    model, metrics = train_and_evaluate(df)
    save_artifacts(model, metrics)
    log.info("Baseline training complete.")
    log.info(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
