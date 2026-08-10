"""
Trains two models from logged windows in DynamoDB:

  1. Directional model: given a feature snapshot at any point in a window,
     predict P(settlement_price >= open_price) -- calibrated probability.

  2. Flip model: given a feature snapshot taken during the final 90s of a
     window, predict P(a flip occurs before settlement) -- i.e. price
     crosses the strike again after having settled on one side.

Both use gradient-boosted trees (XGBoost) and walk-forward validation:
NEVER a random split, since these are time-ordered and adjacent windows are
correlated (autocorrelated volatility regimes) -- a random split leaks
future information into training and will look artificially good.

Run as a weekly batch job (cron) on the EC2 box:
    python models/train.py

Requires at least a couple hundred closed windows to produce anything
meaningful; will warn and exit early if there isn't enough data yet.
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

from dynamo_client import list_closed_windows, get_window_ticks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("MODEL_BUCKET", "btc-kalshi-model-artifacts")
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
MIN_WINDOWS_REQUIRED = 200

FEATURE_COLS = [
    "feat_momentum_5s", "feat_momentum_15s", "feat_momentum_60s",
    "feat_realized_vol_5s", "feat_realized_vol_15s", "feat_realized_vol_60s",
    "feat_acceleration", "feat_book_imbalance", "feat_spread",
    "feat_depth_thinning_10s", "feat_distance_to_strike_stdevs",
    "feat_log_return_from_strike", "feat_seconds_remaining",
]


def load_dataset() -> pd.DataFrame:
    windows = list_closed_windows(region_name=REGION)
    log.info(f"Found {len(windows)} closed windows in btc_windows")
    if len(windows) < MIN_WINDOWS_REQUIRED:
        log.warning(
            f"Only {len(windows)} windows available (< {MIN_WINDOWS_REQUIRED} minimum). "
            "Keep collecting data before training -- an undertrained model here "
            "is worse than no model. Exiting."
        )
        sys.exit(0)

    windows = sorted(windows, key=lambda w: w["window_id"])
    rows = []
    for w in windows:
        wid = w["window_id"]
        ticks = get_window_ticks(wid, region_name=REGION)
        if not ticks:
            continue
        outcome_up = 1 if w.get("outcome") == "up" else 0
        flip_occurred = bool(w.get("flip_occurred", False))
        for t in ticks:
            row = {c: float(t[c]) for c in FEATURE_COLS if c in t and t[c] is not None}
            if len(row) < len(FEATURE_COLS) * 0.5:
                continue  # too sparse a snapshot, skip
            row["window_id"] = wid
            row["timestamp"] = float(t["timestamp"])
            row["label_up"] = outcome_up
            row["label_flip"] = int(flip_occurred)
            row["seconds_remaining"] = float(t.get("feat_seconds_remaining", -1))
            rows.append(row)

    df = pd.DataFrame(rows)
    log.info(f"Built dataset: {len(df)} rows across {df['window_id'].nunique()} windows")
    return df


def walk_forward_splits(df: pd.DataFrame, n_splits: int = 4):
    """Time-ordered splits by window_id -- train on earlier windows, test on
    the next chronological chunk. Never shuffle."""
    window_ids = sorted(df["window_id"].unique())
    fold_size = max(1, len(window_ids) // (n_splits + 1))
    for i in range(1, n_splits + 1):
        train_ids = set(window_ids[: i * fold_size])
        test_ids = set(window_ids[i * fold_size: (i + 1) * fold_size])
        if not test_ids:
            continue
        train_df = df[df["window_id"].isin(train_ids)]
        test_df = df[df["window_id"].isin(test_ids)]
        yield train_df, test_df


def train_directional_model(df: pd.DataFrame):
    X_cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=X_cols + ["label_up"])

    aucs, briers = [], []
    for train_df, test_df in walk_forward_splits(df):
        if train_df.empty or test_df.empty:
            continue
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        )
        model.fit(train_df[X_cols], train_df["label_up"])
        preds = model.predict_proba(test_df[X_cols])[:, 1]
        if test_df["label_up"].nunique() > 1:
            aucs.append(roc_auc_score(test_df["label_up"], preds))
        briers.append(brier_score_loss(test_df["label_up"], preds))

    log.info(f"[Directional] walk-forward AUC: {np.mean(aucs):.4f} | Brier: {np.mean(briers):.4f}")
    log.info("Brier score is the one that matters for EV math -- lower is better "
              "calibrated. AUC alone can look fine while probabilities are still miscalibrated.")

    final_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    )
    final_model.fit(df[X_cols], df["label_up"])
    return final_model, {"auc": float(np.mean(aucs)) if aucs else None, "brier": float(np.mean(briers))}


def train_flip_model(df: pd.DataFrame):
    """Trained only on rows within the final 90s of a window -- the flip
    question is only meaningful once you're in (or near) the settlement
    averaging period."""
    df = df[df["seconds_remaining"] <= 90]
    X_cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=X_cols + ["label_flip"])

    if df["window_id"].nunique() < 20:
        log.warning("Not enough late-window rows to train a flip model yet. Skipping.")
        return None, {}

    aucs, briers = [], []
    for train_df, test_df in walk_forward_splits(df, n_splits=3):
        if train_df.empty or test_df.empty:
            continue
        model = xgb.XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        )
        model.fit(train_df[X_cols], train_df["label_flip"])
        preds = model.predict_proba(test_df[X_cols])[:, 1]
        if test_df["label_flip"].nunique() > 1:
            aucs.append(roc_auc_score(test_df["label_flip"], preds))
        briers.append(brier_score_loss(test_df["label_flip"], preds))

    log.info(f"[Flip] walk-forward AUC: {np.mean(aucs) if aucs else float('nan'):.4f} | "
              f"Brier: {np.mean(briers):.4f}")

    final_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    )
    final_model.fit(df[X_cols], df["label_flip"])
    return final_model, {"auc": float(np.mean(aucs)) if aucs else None, "brier": float(np.mean(briers))}


def save_artifacts(directional_model, flip_model, metrics: dict, feature_cols: list[str]):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    directional_model.save_model(os.path.join(ARTIFACT_DIR, "directional_model.json"))
    if flip_model is not None:
        flip_model.save_model(os.path.join(ARTIFACT_DIR, "flip_model.json"))
    with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
        json.dump({
            "trained_at": datetime.utcnow().isoformat(),
            "feature_cols": feature_cols,
            "metrics": metrics,
        }, f, indent=2)

    try:
        s3 = boto3.client("s3", region_name=REGION)
        for fname in ["directional_model.json", "flip_model.json", "metadata.json"]:
            local_path = os.path.join(ARTIFACT_DIR, fname)
            if os.path.exists(local_path):
                s3.upload_file(local_path, S3_BUCKET, f"models/{fname}")
        log.info(f"Uploaded artifacts to s3://{S3_BUCKET}/models/")
    except Exception as e:
        log.warning(f"S3 upload failed (continuing -- local artifacts still saved): {e}")


def main():
    df = load_dataset()
    directional_model, dir_metrics = train_directional_model(df)
    flip_model, flip_metrics = train_flip_model(df)
    save_artifacts(directional_model, flip_model,
                    {"directional": dir_metrics, "flip": flip_metrics},
                    [c for c in FEATURE_COLS if c in df.columns])
    log.info("Training complete.")


if __name__ == "__main__":
    main()
