"""
Trains two models from logged windows in DynamoDB:

  1. Directional model: given a feature snapshot at any point in a window,
     predict P(settlement_price >= open_price) -- calibrated probability.

  2. Flip model: given a feature snapshot taken during the final 90s of a
     window, predict P(a flip occurs before settlement) -- i.e. price
     crosses the strike again after having settled on one side.

MULTI-EXCHANGE: pulls Coinbase (primary tick grid), Kraken, and Crypto.com
ticks for each window, aligning Kraken's and Crypto.com's feature snapshots
onto Coinbase's tick timestamps via an as-of merge (same technique already
validated in lead_lag_analysis.py / convergence_analysis.py). Each
exchange's features get a prefix (cb_, kr_, cc_) so the model can use all
three without column collisions, and so feature importance later tells you
which exchange's signal actually matters.

A merge tolerance caps how stale a cross-exchange match can be -- this
matters specifically because Crypto.com's public trade tape has been
observed to be bursty/sparse (long gaps between prints); without a
tolerance, a Crypto.com feature snapshot from minutes ago could get
silently paired with a current Coinbase tick, which would be actively
misleading rather than just missing. Beyond the tolerance, those columns
are NaN for that row rather than using stale data.

Windows that predate an exchange's ingestion start (or backfilled windows,
which have no tick data for ANY exchange) will simply have NaN in that
exchange's columns for those rows -- dropna during training handles this,
at the cost of using fewer rows from the earliest period.

Both models use gradient-boosted trees (XGBoost) and walk-forward
validation: NEVER a random split, since these are time-ordered and
adjacent windows are correlated (autocorrelated volatility regimes) -- a
random split leaks future information into training and will look
artificially good.

Run as a periodic batch job (cron) on the EC2 box:
    python train.py

Requires at least a couple hundred closed windows with real (non-backfill)
tick data to produce anything meaningful; will warn and exit early if
there isn't enough data yet.
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
from boto3.dynamodb.conditions import Key

from dynamo_client import list_closed_windows, get_window_ticks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("MODEL_BUCKET", "btc-kalshi-model-artifacts")
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
MIN_WINDOWS_REQUIRED = 200
CROSS_EXCHANGE_MERGE_TOLERANCE_SEC = 20.0

BASE_FEATURES = [
    "momentum_5s", "momentum_15s", "momentum_60s",
    "realized_vol_5s", "realized_vol_15s", "realized_vol_60s",
    "acceleration", "book_imbalance", "spread", "depth_thinning_10s",
    "distance_to_strike_stdevs", "log_return_from_strike",
    "local_extrema_60s", "path_curvature_60s", "ma_cross_count_60s",
    "recent_range_ratio", "imbalance_5", "imbalance_20", "imbalance_50",
    "spread_change_rate_10s", "spread_zscore_60s",
    "spread_local_extrema_60s", "spread_curvature_60s", "spread_ma_cross_60s",
]

EXCHANGE_PREFIXES = {"cb": "btc_ticks", "kr": "kraken_ticks", "cc": "cryptocom_ticks"}

FEATURE_COLS = [f"{prefix}_{feat}" for prefix in EXCHANGE_PREFIXES for feat in BASE_FEATURES]

CONSENSUS_METRICS = ["book_imbalance", "imbalance_5", "imbalance_20", "imbalance_50"]
CONSENSUS_FEATURE_COLS = [
    f"consensus_{metric}_{suffix}"
    for metric in CONSENSUS_METRICS
    for suffix in ["n_positive", "n_negative", "all_agree", "mean", "dispersion"]
]
FEATURE_COLS = FEATURE_COLS + CONSENSUS_FEATURE_COLS


def get_generic_window_ticks(table_name: str, window_id: str, region: str = REGION) -> list[dict]:
    dynamodb = boto3.resource("dynamodb", region_name=region)
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


def _ticks_to_feature_df(ticks: list[dict], prefix: str) -> pd.DataFrame | None:
    if not ticks:
        return None
    rows = []
    for t in ticks:
        row = {"timestamp": float(t["timestamp"])}
        for feat in BASE_FEATURES:
            key = f"feat_{feat}"
            if key in t and t[key] is not None:
                row[f"{prefix}_{feat}"] = float(t[key])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("timestamp")
    return df if len(df) > 0 else None


def build_window_dataframe(window_id: str, outcome_up: int, flip_occurred: bool, region: str = REGION) -> pd.DataFrame | None:
    cb_ticks = get_generic_window_ticks("btc_ticks", window_id, region)
    cb_df = _ticks_to_feature_df(cb_ticks, "cb")
    if cb_df is None:
        return None

    kr_ticks = get_generic_window_ticks("kraken_ticks", window_id, region)
    kr_df = _ticks_to_feature_df(kr_ticks, "kr")

    cc_ticks = get_generic_window_ticks("cryptocom_ticks", window_id, region)
    cc_df = _ticks_to_feature_df(cc_ticks, "cc")

    merged = cb_df
    for other_df in (kr_df, cc_df):
        if other_df is not None:
            merged = pd.merge_asof(
                merged, other_df, on="timestamp", direction="backward",
                tolerance=CROSS_EXCHANGE_MERGE_TOLERANCE_SEC,
            )

    merged["window_id"] = window_id
    merged["label_up"] = outcome_up
    merged["label_flip"] = int(flip_occurred)
    cb_seconds_remaining = {float(t["timestamp"]): float(t.get("feat_seconds_remaining", -1)) for t in cb_ticks}
    merged["seconds_remaining"] = merged["timestamp"].map(cb_seconds_remaining)

    merged = add_consensus_features(merged)

    return merged


def add_consensus_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-exchange agreement/disagreement features -- captures patterns
    like 'all three books lean the same way' vs 'exchanges disagree',
    which individual per-exchange columns don't directly expose to the
    model even though a tree could in principle learn the interaction on
    its own. Computed per metric (book_imbalance and each depth tier)."""
    df = df.copy()
    for metric in ["book_imbalance", "imbalance_5", "imbalance_20", "imbalance_50"]:
        cols = [f"{prefix}_{metric}" for prefix in EXCHANGE_PREFIXES if f"{prefix}_{metric}" in df.columns]
        if len(cols) < 2:
            continue  # need at least 2 exchanges present to compute agreement

        vals = df[cols]
        n_available = vals.notna().sum(axis=1)
        n_positive = (vals > 0).sum(axis=1)
        n_negative = (vals < 0).sum(axis=1)

        df[f"consensus_{metric}_n_positive"] = n_positive
        df[f"consensus_{metric}_n_negative"] = n_negative
        # all_agree: every AVAILABLE exchange has the same sign (only
        # meaningful when at least 2 are available; NaN otherwise so it
        # doesn't get misread as "disagreement" when data's just missing)
        all_agree = np.where(
            n_available >= 2,
            ((n_positive == n_available) | (n_negative == n_available)).astype(float),
            np.nan,
        )
        df[f"consensus_{metric}_all_agree"] = all_agree
        df[f"consensus_{metric}_mean"] = vals.mean(axis=1, skipna=True)
        df[f"consensus_{metric}_dispersion"] = vals.std(axis=1, skipna=True)

    return df


def load_dataset() -> pd.DataFrame:
    windows = list_closed_windows(region_name=REGION)
    windows = [w for w in windows if w.get("source") != "backfill"]
    log.info(f"Found {len(windows)} closed non-backfill windows in btc_windows")
    if len(windows) < MIN_WINDOWS_REQUIRED:
        log.warning(
            f"Only {len(windows)} windows available (< {MIN_WINDOWS_REQUIRED} minimum). "
            "Keep collecting data before training -- an undertrained model here "
            "is worse than no model. Exiting."
        )
        sys.exit(0)

    windows = sorted(windows, key=lambda w: w["window_id"])
    all_dfs = []
    for w in windows:
        wid = w["window_id"]
        outcome_up = 1 if w.get("outcome") == "up" else 0
        flip_occurred = bool(w.get("flip_occurred", False))
        window_df = build_window_dataframe(wid, outcome_up, flip_occurred, region=REGION)
        if window_df is not None:
            all_dfs.append(window_df)

    if not all_dfs:
        log.warning("No usable windows with tick data found. Exiting.")
        sys.exit(0)

    df = pd.concat(all_dfs, ignore_index=True)
    log.info(f"Built dataset: {len(df)} rows across {df['window_id'].nunique()} windows")

    for prefix, table_name in EXCHANGE_PREFIXES.items():
        cols_present = [c for c in FEATURE_COLS if c.startswith(f"{prefix}_") and c in df.columns]
        coverage = df[cols_present].notna().any(axis=1).mean() if cols_present else 0.0
        log.info(f"  {table_name} ({prefix}_*): {coverage*100:.1f}% of rows have at least one non-null feature")

    return df


def walk_forward_splits(df: pd.DataFrame, n_splits: int = 4):
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
    log.info(f"[Directional] {len(df)} rows remain after dropping incomplete cross-exchange rows")

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

    importances = dict(zip(X_cols, final_model.feature_importances_.tolist()))
    sorted_importances = dict(sorted(importances.items(), key=lambda x: -x[1])[:15])
    log.info("Top 15 feature importances:")
    for k, v in sorted_importances.items():
        log.info(f"  {k}: {v:.4f}")

    return final_model, {
        "auc": float(np.mean(aucs)) if aucs else None,
        "brier": float(np.mean(briers)),
        "top_feature_importances": sorted_importances,
    }


def train_flip_model(df: pd.DataFrame):
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
