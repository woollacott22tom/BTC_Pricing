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
    "distance_to_round_100", "distance_to_round_1000",
    "volume_15s", "volume_60s",
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

# Kalshi's OWN trailing price momentum -- without this, the model has no
# way to learn "this order-book signal means continuation when Kalshi is
# already trending, but reversal when it isn't", since it never sees
# whether Kalshi was trending in the first place. A tree-based model can
# learn this kind of conditional interaction automatically GIVEN the
# feature is present -- no need to hand-classify continuation vs reversal
# the way the standalone event-study script does.
KALSHI_MOMENTUM_LOOKBACKS = {"kalshi_momentum_15s": 15.0, "kalshi_momentum_60s": 60.0}
FEATURE_COLS = FEATURE_COLS + list(KALSHI_MOMENTUM_LOOKBACKS.keys())
FEATURE_COLS = FEATURE_COLS + ["hour_of_day_utc", "day_of_week"]

# Reversal label config: "is Kalshi's price about to reverse against its
# current trend, at ANY point in the window" -- not the old settlement-
# specific flip concept (window-level, final-60s-only). This is the same
# underlying question asked continuously throughout the window, usable for
# both entry (imminent trend forming) and exit (imminent trend breaking)
# decisions. See train_reversal_model() and compute_reversal_labels().
REVERSAL_HORIZON_SEC = 45.0  # matches the horizon validated against Kalshi's 15s API cache
REVERSAL_THRESHOLD_CENTS = 3.0  # minimum forward move to count as a genuine reversal, not noise
KALSHI_MERGE_TOLERANCE_SEC = 20.0  # Kalshi's own API caches responses for 15s server-side
                                     # (confirmed via kalshi_polling_diagnostic.py) -- a tick
                                     # older than that is genuinely stale, not just missing

# Before this timestamp, all three exchanges' ingestion services logged
# ticks ONLY when a trade fired -- book-only movement between trades (the
# actual mechanism behind several patterns discussed: Coinbase's book
# gradually flipping through many small fills during a slow move vs.
# Crypto.com sitting negative until one large trade forces a snap
# confirmation) was invisible to storage. Order-book-derived features and
# volume from before this cutoff aren't just sparser -- they're shaped
# wrong, having been forced through the same coarse, trade-gated sampling
# regardless of which exchange actually produced them, which can flatten
# or hide the exact asymmetry between exchanges that makes them useful.
# Price/momentum features are NOT excluded here -- past discussion
# concluded that gap is real but much less consequential than the
# book-state one.
#
# This uses the LATER of two separate deploys (Coinbase + Kraken fix) as
# a single, conservative, uniform cutoff for all three exchanges -- the
# Crypto.com fix landed separately and earlier, but that exact timestamp
# wasn't confirmed with certainty, so the safer later cutoff is used
# everywhere rather than guessing at an earlier one.
ORDER_BOOK_FIX_CUTOFF_TS = 1787118936.0  # 2026-08-19T05:55:36+00:00 UTC

ORDER_BOOK_AND_VOLUME_FEATURES = {
    "book_imbalance", "imbalance_5", "imbalance_20", "imbalance_50",
    "spread", "spread_change_rate_10s", "spread_zscore_60s",
    "spread_local_extrema_60s", "spread_curvature_60s", "spread_ma_cross_60s",
    "depth_thinning_10s", "volume_15s", "volume_60s",
}

# Cross-exchange price gap features -- does one exchange trade at a
# persistent premium/discount to another, and is the CURRENT gap wider or
# narrower than its own recent typical level? The raw dollar gap alone
# isn't very informative on its own (a persistent baseline offset carries
# little predictive signal by itself), but DEVIATION from that baseline
# is: it isolates "the gap is unusual right now" from "there's always some
# gap", matching the specific pattern of a gap widening during a price
# move and reverting once the move settles.
PRICE_DIFF_PAIRS = [("cc", "cb"), ("kr", "cb")]
# Multiple rolling baselines, not just one -- a single hand-picked window
# (e.g. 60s) bakes in a specific guess about the right timescale. Giving
# the model several windows lets it discover which one (if any) actually
# carries signal, rather than betting everything on one assumption drawn
# from limited manual observation.
PRICE_DIFF_ROLLING_BASELINE_SECS = [30.0, 60.0, 120.0]
PRICE_DIFF_FEATURE_COLS = []
for _other, _base in PRICE_DIFF_PAIRS:
    PRICE_DIFF_FEATURE_COLS.append(f"price_diff_{_other}_{_base}")
    for _window in PRICE_DIFF_ROLLING_BASELINE_SECS:
        PRICE_DIFF_FEATURE_COLS.append(f"price_diff_{_other}_{_base}_deviation_{int(_window)}s")
FEATURE_COLS = FEATURE_COLS + PRICE_DIFF_FEATURE_COLS


def add_price_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds raw cross-exchange price gaps and their deviation from SEVERAL
    trailing rolling baselines (not just one -- see PRICE_DIFF_ROLLING_BASELINE_SECS
    comment). Uses TIME-based rolling windows (not fixed row counts), which
    only look backward from each row's own timestamp -- no future leakage,
    correct regardless of how tick frequency varies between exchanges."""
    df = df.copy()
    df["_dt"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("_dt")

    for other, base in PRICE_DIFF_PAIRS:
        other_col, base_col = f"{other}_price", f"{base}_price"
        if other_col not in df.columns or base_col not in df.columns:
            continue
        diff_col = f"price_diff_{other}_{base}"
        df[diff_col] = df[other_col] - df[base_col]
        for window in PRICE_DIFF_ROLLING_BASELINE_SECS:
            rolling_mean = df[diff_col].rolling(f"{int(window)}s", min_periods=3).mean()
            df[f"{diff_col}_deviation_{int(window)}s"] = df[diff_col] - rolling_mean

    return df.reset_index(drop=True)


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
        if "price" in t and t["price"] is not None:
            row[f"{prefix}_price"] = float(t["price"])
        for feat in BASE_FEATURES:
            key = f"feat_{feat}"
            if key in t and t[key] is not None:
                row[f"{prefix}_{feat}"] = float(t[key])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("timestamp")
    return df if len(df) > 0 else None


def _kalshi_ticks_to_momentum_df(kalshi_ticks: list[dict]) -> pd.DataFrame | None:
    """Builds Kalshi's own trailing momentum at each of ITS OWN tick
    timestamps -- momentum_Ns(t) = price change over the N seconds up to
    and including t, using ONLY prior observations (never future ones,
    which would leak the label). This gets merge_asof'd onto Coinbase's
    tick grid the same way the other exchanges are, so it's just another
    input feature from the model's point of view."""
    if not kalshi_ticks:
        return None
    rows = []
    for t in kalshi_ticks:
        if "yes_mid_cents" not in t or t["yes_mid_cents"] is None:
            continue
        rows.append({"timestamp": float(t["timestamp"]), "yes_mid_cents": float(t["yes_mid_cents"])})
    if len(rows) < 2:
        return None

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    for col_name, lookback in KALSHI_MOMENTUM_LOOKBACKS.items():
        momentum = []
        for i in range(len(df)):
            t_now = df["timestamp"].iloc[i]
            # only rows STRICTLY BEFORE or AT t_now -- no future leakage
            past = df[(df["timestamp"] <= t_now) & (df["timestamp"] >= t_now - lookback)]
            if len(past) < 2:
                momentum.append(np.nan)
            else:
                momentum.append(past["yes_mid_cents"].iloc[-1] - past["yes_mid_cents"].iloc[0])
        df[col_name] = momentum

    return df[["timestamp"] + list(KALSHI_MOMENTUM_LOOKBACKS.keys())]


def _kalshi_ticks_to_price_df(kalshi_ticks: list[dict]) -> pd.DataFrame | None:
    """Raw Kalshi price series (NOT momentum) -- used only for computing
    the forward-looking reversal LABEL. Labels are allowed to look into
    the future (that's what makes them a trainable target); this must
    stay completely separate from the momentum FEATURE computation above,
    which must never see the future."""
    if not kalshi_ticks:
        return None
    rows = []
    for t in kalshi_ticks:
        if "yes_mid_cents" not in t or t["yes_mid_cents"] is None:
            continue
        rows.append({"timestamp": float(t["timestamp"]), "yes_mid_cents": float(t["yes_mid_cents"])})
    if len(rows) < 2:
        return None
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def compute_reversal_labels(grid_df: pd.DataFrame, kalshi_price_df: pd.DataFrame | None,
                             horizon_seconds: float = REVERSAL_HORIZON_SEC,
                             threshold_cents: float = REVERSAL_THRESHOLD_CENTS,
                             tolerance_seconds: float = KALSHI_MERGE_TOLERANCE_SEC) -> pd.Series:
    """For each row in grid_df (must already have 'timestamp' and
    'kalshi_momentum_15s' columns), determines whether Kalshi's price
    REVERSES against its current trailing direction within the next
    horizon_seconds, by at least threshold_cents.

      1 = genuine reversal (price moved the OPPOSITE way from the current
          trend, by more than the noise threshold)
      0 = no reversal (price continued the same direction, or moved by
          less than the threshold either way)
      NaN = excluded -- either there's no clear existing trend to reverse
          FROM (momentum ~0), or forward data isn't available within
          tolerance (e.g. near the very end of the window)

    Uses the SAME vectorized merge_asof technique as everything else here,
    just with a FORWARD-looking target this time -- fine for a label,
    which is explicitly the case for a supervised training TARGET, as
    opposed to a feature (which must never see the future)."""
    n = len(grid_df)
    if kalshi_price_df is None or "kalshi_momentum_15s" not in grid_df.columns:
        return pd.Series([np.nan] * n, index=grid_df.index)

    baseline_df = pd.merge_asof(
        grid_df[["timestamp"]], kalshi_price_df, on="timestamp", direction="backward",
        tolerance=tolerance_seconds,
    )
    baseline_price = baseline_df["yes_mid_cents"].reset_index(drop=True)

    target_lookup = grid_df[["timestamp"]].copy()
    target_lookup["target_ts"] = target_lookup["timestamp"] + horizon_seconds
    target_lookup = target_lookup.sort_values("target_ts")
    target_df = pd.merge_asof(
        target_lookup.rename(columns={"timestamp": "orig_ts", "target_ts": "timestamp"}),
        kalshi_price_df, on="timestamp", direction="backward", tolerance=tolerance_seconds,
    )
    target_df = target_df.sort_values("orig_ts").reset_index(drop=True)
    target_price = target_df["yes_mid_cents"]

    forward_change = target_price - baseline_price
    current_direction = np.sign(grid_df["kalshi_momentum_15s"].reset_index(drop=True))

    label = pd.Series([np.nan] * n)
    valid = forward_change.notna() & current_direction.notna() & (current_direction != 0)
    reversed_mask = valid & (np.sign(forward_change) != current_direction) & (forward_change.abs() >= threshold_cents)
    not_reversed_mask = valid & ~reversed_mask

    label[reversed_mask.to_numpy()] = 1
    label[not_reversed_mask.to_numpy()] = 0
    label.index = grid_df.index
    return label


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

    merged = add_price_differential_features(merged)

    kalshi_ticks = get_generic_window_ticks("kalshi_prices", window_id, region)
    kalshi_momentum_df = _kalshi_ticks_to_momentum_df(kalshi_ticks)
    if kalshi_momentum_df is not None:
        merged = pd.merge_asof(
            merged, kalshi_momentum_df, on="timestamp", direction="backward",
            tolerance=KALSHI_MERGE_TOLERANCE_SEC,
        )

    merged["window_id"] = window_id
    merged["label_up"] = outcome_up
    merged["label_flip"] = int(flip_occurred)  # legacy: settlement-specific flip, window-level only
    cb_seconds_remaining = {float(t["timestamp"]): float(t.get("feat_seconds_remaining", -1)) for t in cb_ticks}
    merged["seconds_remaining"] = merged["timestamp"].map(cb_seconds_remaining)

    # General-purpose "is a reversal imminent right now" label -- usable at
    # ANY point in the window, not just the settlement tail. This is what
    # actually serves both entering on an expected trend AND exiting
    # before an expected flip, since it's the same underlying question.
    kalshi_price_df = _kalshi_ticks_to_price_df(kalshi_ticks)
    merged["label_reversal"] = compute_reversal_labels(merged, kalshi_price_df)

    # Raw time-of-day/day-of-week -- NOT assuming which hours are "high
    # volatility"/"institutional" a priori. Letting the model (and separate
    # analysis) discover empirically whether specific hours show stronger
    # signal, rather than hand-guessing US market hours or similar.
    merged["hour_of_day_utc"] = pd.to_datetime(merged["timestamp"], unit="s").dt.hour
    merged["day_of_week"] = pd.to_datetime(merged["timestamp"], unit="s").dt.dayofweek

    merged = add_consensus_features(merged)

    # Drop pre-cutoff ticks entirely -- see ORDER_BOOK_FIX_CUTOFF_TS comment
    # above. NOTE: this is the simple, safe version -- it drops the WHOLE
    # row, which also discards otherwise-still-valid price/momentum data
    # from those same moments, not just the untrusted order-book/volume
    # columns. A more surgical version (NaN only the untrusted columns,
    # keep the row, rely on XGBoost's native missing-value handling) is
    # possible but bigger and untested -- worth building if the row-level
    # cut turns out too costly once there's enough post-cutoff data to
    # compare against.
    before_cutoff = len(merged)
    merged = merged[merged["timestamp"] >= ORDER_BOOK_FIX_CUTOFF_TS].reset_index(drop=True)
    if before_cutoff > len(merged):
        merged.attrs["rows_dropped_pre_cutoff"] = before_cutoff - len(merged)

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
    total_dropped_pre_cutoff = 0
    for w in windows:
        wid = w["window_id"]
        # Prefer Kalshi's own authoritative settlement result (written by
        # kalshi_settlement_reconciliation.py) over our own Coinbase-TWAP
        # approximation whenever it's available -- Kalshi's real outcome
        # is ground truth, our own calculation is only an approximation
        # of it. Falls back to our own outcome for windows that haven't
        # been reconciled yet (e.g. very recent ones, or if the
        # reconciliation script hasn't been run since).
        true_outcome = w.get("kalshi_true_outcome")
        outcome_up = 1 if (true_outcome or w.get("outcome")) == "up" else 0
        flip_occurred = bool(w.get("flip_occurred", False))
        window_df = build_window_dataframe(wid, outcome_up, flip_occurred, region=REGION)
        if window_df is not None:
            total_dropped_pre_cutoff += window_df.attrs.get("rows_dropped_pre_cutoff", 0)
            all_dfs.append(window_df)

    if not all_dfs:
        log.warning("No usable windows with tick data found. Exiting.")
        sys.exit(0)

    df = pd.concat(all_dfs, ignore_index=True)
    log.info(f"Built dataset: {len(df)} rows across {df['window_id'].nunique()} windows")
    if total_dropped_pre_cutoff > 0:
        cutoff_dt = datetime.utcfromtimestamp(ORDER_BOOK_FIX_CUTOFF_TS).isoformat() + "Z"
        log.info(f"Dropped {total_dropped_pre_cutoff} pre-cutoff rows (before {cutoff_dt}) -- "
                 f"order-book/volume data from before the book-logging fix was excluded as unreliable.")

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


def train_reversal_model(df: pd.DataFrame):
    """Replaces the old settlement-only flip model. Trained on ALL rows
    across the entire window (no seconds_remaining filter), predicting
    whether Kalshi's price is about to reverse against its CURRENT
    trailing direction -- usable continuously throughout the window, for
    both entering on an expected move and exiting before an expected
    reversal. Same underlying question either way, just applied at
    whatever point in the window you check it.

    Saved as flip_model.json for continuity with app.py's existing
    serving hook -- what changed is what the model predicts, not where
    the artifact lives."""
    X_cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=X_cols + ["label_reversal"])
    log.info(f"[Reversal] {len(df)} rows remain after dropping rows with no clear "
              f"existing trend or missing forward Kalshi data")

    if df["window_id"].nunique() < 20:
        log.warning("Not enough rows with a usable reversal label yet. Skipping.")
        return None, {}

    aucs, briers = [], []
    for train_df, test_df in walk_forward_splits(df, n_splits=3):
        if train_df.empty or test_df.empty:
            continue
        model = xgb.XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        )
        model.fit(train_df[X_cols], train_df["label_reversal"])
        preds = model.predict_proba(test_df[X_cols])[:, 1]
        if test_df["label_reversal"].nunique() > 1:
            aucs.append(roc_auc_score(test_df["label_reversal"], preds))
        briers.append(brier_score_loss(test_df["label_reversal"], preds))

    log.info(f"[Reversal] walk-forward AUC: {np.mean(aucs) if aucs else float('nan'):.4f} | "
              f"Brier: {np.mean(briers):.4f}")

    final_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    )
    final_model.fit(df[X_cols], df["label_reversal"])
    return final_model, {"auc": float(np.mean(aucs)) if aucs else None, "brier": float(np.mean(briers))}


def save_artifacts(directional_model, reversal_model, metrics: dict, feature_cols: list[str]):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    directional_model.save_model(os.path.join(ARTIFACT_DIR, "directional_model.json"))
    if reversal_model is not None:
        reversal_model.save_model(os.path.join(ARTIFACT_DIR, "flip_model.json"))
    with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
        json.dump({
            "trained_at": datetime.utcnow().isoformat(),
            "feature_cols": feature_cols,
            "metrics": metrics,
            "note": "flip_model.json now holds the general-purpose reversal model "
                    "(usable at any point in the window), not the old settlement-only flip model.",
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
    reversal_model, reversal_metrics = train_reversal_model(df)
    save_artifacts(directional_model, reversal_model,
                    {"directional": dir_metrics, "reversal": reversal_metrics},
                    [c for c in FEATURE_COLS if c in df.columns])
    log.info("Training complete.")


if __name__ == "__main__":
    main()
