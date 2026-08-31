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
import re
import sys
import gc
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

# Strike-cross label config: "does Coinbase's price actually cross the
# window's strike at any point BEFORE THE WINDOW CLOSES" -- the most
# directly trade-relevant of the three trained questions, since crossing
# the strike is literally the event that flips which side of the Kalshi
# contract wins. Distinct from BOTH the (scrapped) directional/settlement
# model (which asked where price would SETTLE) and the reversal model
# (which asks whether MOMENTUM reverses, with no notion of the strike at
# all).
#
# No fixed forward horizon (e.g. "within the next 45s") -- "before close"
# uses the window's own natural endpoint instead of guessing at how long a
# crossing takes to develop. Real-world observation: crossings are rare on
# a scale of seconds and typically take a few minutes, so an arbitrary
# fixed horizon risks either missing genuine crossings (too short) or
# needing constant re-tuning (too long, or the wrong length for different
# market conditions). Also detects a crossing even if price later flips
# BACK before close -- "did it ever cross," not just "which side does it
# end up on."
#
# Needs no new stored data: the sign of the already-stored
# cb_log_return_from_strike column IS which side of the strike price is
# on (positive = above, negative = below), so a crossing is exactly a
# sign flip of that column at any point between now and the window's last
# row -- no separate numeric strike value needs to be threaded through.

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


def compute_strike_cross_labels(grid_df: pd.DataFrame) -> pd.Series:
    """For each row in grid_df (must already have 'cb_log_return_from_strike'),
    determines whether Coinbase's price crosses the window's strike at ANY
    point between now and the window's close -- not a fixed forward
    horizon. Detects a crossing even if price later flips back before
    close ("did it ever cross," not just "which side does it end up on").

    Computed via a reverse-cumulative min/max of cb_log_return_from_strike
    from each row through the end of the window: if currently above the
    strike (positive), a crossing happens iff the value ever goes negative
    somewhere between now and close (forward min < 0); if currently below
    (negative), iff it ever goes positive (forward max > 0). No horizon or
    tolerance parameter needed -- "before close" IS the horizon, using the
    window's own natural endpoint. Since grid_df is built one window at a
    time (see build_window_dataframe), this naturally never leaks into a
    different window's data.

      1 = crosses the strike at some point before close
      0 = stays on the same side all the way to close (including the
          trivial last row, which has no more time left to flip)
      NaN = current value is exactly 0 (on the strike itself -- ambiguous
          starting side, vanishingly rare with continuous prices)

    Verified against a hand-traced dip-and-recover sequence before
    shipping -- correctly flags a crossing even when price flips back to
    its original side before close, not just checking the final value."""
    n = len(grid_df)
    if "cb_log_return_from_strike" not in grid_df.columns:
        return pd.Series([np.nan] * n, index=grid_df.index)

    vals = grid_df["cb_log_return_from_strike"].reset_index(drop=True).to_numpy()
    current_side = np.sign(vals)

    forward_min = np.minimum.accumulate(vals[::-1])[::-1]
    forward_max = np.maximum.accumulate(vals[::-1])[::-1]

    label = pd.Series([np.nan] * n)
    above_now = current_side > 0
    below_now = current_side < 0

    label[(above_now & (forward_min < 0))] = 1
    label[(above_now & (forward_min >= 0))] = 0
    label[(below_now & (forward_max > 0))] = 1
    label[(below_now & (forward_max <= 0))] = 0
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

    # "Does price actually cross the strike within the next horizon" --
    # the most directly trade-relevant of the three trained questions.
    # Computed from cb_log_return_from_strike's sign alone (see
    # compute_strike_cross_labels docstring) -- needs cb_ prefixed columns
    # to already exist in merged, which they do by this point.
    merged["label_strike_cross"] = compute_strike_cross_labels(merged)

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


def _safe_cache_filename(window_id: str) -> str:
    """Sanitizes a window_id (e.g. '2026-08-19T05:45:00+00:00') into a
    safe filename -- avoids relying on colons/plus signs being legal in
    the filesystem, and sidesteps any shell-escaping surprises."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", window_id) + ".pkl"


def load_dataset() -> pd.DataFrame:
    windows = list_closed_windows(region_name=REGION)
    windows = [w for w in windows if w.get("source") != "backfill"]
    log.info(f"Found {len(windows)} closed non-backfill windows in btc_windows")

    # Exclude windows ENTIRELY before the book-logging fix cutoff before
    # ever fetching their tick data -- previously these were fully fetched
    # (4 separate DynamoDB queries each: btc_ticks, kraken_ticks,
    # cryptocom_ticks, kalshi_prices) only to have every single row
    # dropped afterward inside build_window_dataframe(), wasting real
    # fetch time and memory for zero usable rows. Only excludes windows
    # whose FULL 900s span ends at or before the cutoff; the one window
    # that straddles the cutoff boundary is still fetched normally and
    # correctly row-filtered as before (some of its rows are genuinely
    # post-cutoff and usable).
    before_filter_count = len(windows)
    windows = [
        w for w in windows
        if datetime.fromisoformat(w["window_id"]).timestamp() + 900.0 > ORDER_BOOK_FIX_CUTOFF_TS
    ]
    skipped_entirely_pre_cutoff = before_filter_count - len(windows)
    if skipped_entirely_pre_cutoff > 0:
        log.info(f"Excluding {skipped_entirely_pre_cutoff} windows entirely before the "
                 f"book-logging fix cutoff -- never fetching their tick data, since every "
                 f"row would be dropped anyway")

    if len(windows) < MIN_WINDOWS_REQUIRED:
        log.warning(
            f"Only {len(windows)} usable (post-cutoff) windows available (< {MIN_WINDOWS_REQUIRED} minimum). "
            "Keep collecting data before training -- an undertrained model here "
            "is worse than no model. Exiting."
        )
        sys.exit(0)

    windows = sorted(windows, key=lambda w: w["window_id"])

    # Stream each window's dataframe to disk immediately after building it,
    # rather than accumulating every window's full dataframe in a Python
    # list before one final concat -- with 1000+ windows across three
    # exchanges, holding everything in memory simultaneously exhausted this
    # box's RAM (confirmed directly: a real run had to be killed manually
    # after climbing past 1.7GB of swap and threatening to take the live
    # serving process down with it via the OOM killer). Peak memory is now
    # bounded by roughly ONE window's processing overhead at a time, plus
    # the final combined dataset read back from disk -- not the sum of
    # every window's overhead held alive at once.
    #
    # Uses pandas' built-in pickle format rather than Parquet specifically
    # so this doesn't introduce a new dependency (pyarrow) that may not be
    # installed on the box -- pickle round-trips pandas dtypes/NaNs exactly
    # and needs nothing beyond pandas itself.
    #
    # FIXED, persistent location (not a randomly-suffixed tempdir) so an
    # interrupted run -- confirmed directly: a 1966-window run was killed
    # by SSM's execution timeout after processing 1142 of them -- can be
    # RESUMED by a later, separate process invocation rather than starting
    # over from zero. Explicitly NOT using the default /tmp -- confirmed
    # via `mount` that /tmp on this box is tmpfs (RAM-backed), capped at
    # ~460MB. The root filesystem (/dev/nvme0n1p1) is real EBS storage
    # with ~21GB free -- genuinely off-heap, not memory in disguise.
    #
    # Deliberately NOT auto-deleted at the end of a run -- an interrupted
    # run's partial progress needs to survive until the NEXT run picks it
    # back up. Safe to manually clear this directory (rm -rf) if you ever
    # want a guaranteed fully-fresh rebuild, e.g. after changing
    # FEATURE_COLS in a way that would make old cached rows stale.
    # Configurable via TRAIN_CACHE_DIR so the same script works unmodified
    # on the EC2 box (default) or anywhere else with real disk and AWS
    # credentials -- e.g. Colab, which offers far more RAM (~12GB+ on the
    # free tier vs. this box's 916MB) for exactly the kind of heavy batch
    # job training is, without risking the always-on live services that
    # share the EC2 box.
    cache_dir = os.environ.get("TRAIN_CACHE_DIR", "/home/ec2-user/train_window_cache")
    os.makedirs(cache_dir, exist_ok=True)
    log.info(f"Using persistent window cache at {cache_dir} (enables resuming an interrupted run)")

    total_dropped_pre_cutoff = 0
    windows_processed_this_run = 0
    windows_resumed_from_cache = 0

    for idx, w in enumerate(windows):
        wid = w["window_id"]
        out_path = os.path.join(cache_dir, _safe_cache_filename(wid))

        if os.path.exists(out_path):
            # Confirm the cached file is actually readable AND has every
            # column the CURRENT schema expects -- not just "did it read
            # back." A cache built before a schema change (e.g. adding
            # label_strike_cross) would otherwise be silently reused
            # as-is, missing whatever's new, and only fail later --
            # confusingly -- during training or not at all if dropna
            # just quietly drops those rows. Required label columns are
            # checked explicitly here since they're exactly the columns
            # most likely to be added incrementally over time.
            try:
                cached_df = pd.read_pickle(out_path)
                required_cols = {"label_reversal", "label_strike_cross"}
                missing = required_cols - set(cached_df.columns)
                if missing:
                    log.warning(f"Cached file for {wid} predates current schema "
                                 f"(missing {missing}) -- rebuilding it")
                    del cached_df
                else:
                    total_dropped_pre_cutoff += cached_df.attrs.get("rows_dropped_pre_cutoff", 0)
                    del cached_df
                    windows_resumed_from_cache += 1
                    continue
            except Exception as e:
                log.warning(f"Cached file for {wid} failed to read back ({e}) -- rebuilding it")

        # Prefer Kalshi's own authoritative settlement result (written by
        # kalshi_settlement_reconciliation.py) over our own Coinbase-TWAP
        # approximation whenever it's available -- Kalshi's real outcome
        # is ground truth, our own calculation is only an approximation
        # of it. Falls back to our own outcome for windows that haven't
        # been reconciled yet.
        true_outcome = w.get("kalshi_true_outcome")
        outcome_up = 1 if (true_outcome or w.get("outcome")) == "up" else 0
        flip_occurred = bool(w.get("flip_occurred", False))
        window_df = build_window_dataframe(wid, outcome_up, flip_occurred, region=REGION)

        if window_df is not None and len(window_df) > 0:
            total_dropped_pre_cutoff += window_df.attrs.get("rows_dropped_pre_cutoff", 0)
            window_df.to_pickle(out_path)
            windows_processed_this_run += 1

        del window_df  # prompt release of this window's processing overhead
                         # before starting the next one, rather than letting
                         # it linger alongside everything still to come

        # Periodic explicit GC -- doesn't force numpy/pandas' underlying
        # C allocators to hand memory back to the OS (they typically
        # retain freed blocks for reuse rather than releasing them), so
        # this won't necessarily show up as falling RSS in `free -h`.
        # What it DOES do is catch reference cycles that pure
        # refcounting can miss, keeping peak growth bounded rather than
        # silently accumulating over a run spanning 1000+ windows.
        if idx % 50 == 0:
            gc.collect()

    log.info(f"{windows_resumed_from_cache} windows resumed from a prior run's cache, "
             f"{windows_processed_this_run} newly processed this run")

    cached_files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".pkl")]
    if not cached_files:
        log.warning("No usable windows with tick data found. Exiting.")
        sys.exit(0)

    # NOTE: this step still reads every cached window's dataframe back into
    # memory at once to build the final combined dataset -- resume avoids
    # REPEATING the expensive per-window DynamoDB fetch/feature-computation
    # work across separate runs, but does not by itself reduce this final
    # step's peak memory. That remains a separate, real concern worth
    # addressing directly if it becomes the next bottleneck, e.g. by
    # training on chunked reads instead of one full in-memory concat.
    log.info(f"Reading back {len(cached_files)} cached window dataframes for training...")
    df = pd.concat([pd.read_pickle(f) for f in cached_files], ignore_index=True)

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


def train_strike_cross_model(df: pd.DataFrame):
    """The most directly trade-relevant of the three trained questions:
    does price actually cross the window's strike within the next
    horizon_seconds, checked continuously throughout the window (same
    usability pattern as the reversal model -- not settlement-only).

    Strike crossings are plausibly rarer than reversals (a reversal is
    just a momentum wobble; a crossing is a bigger move), so class balance
    is checked explicitly and scale_pos_weight applied if warranted --
    unlike the reversal model, which didn't need this in practice."""
    X_cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=X_cols + ["label_strike_cross"])
    log.info(f"[StrikeCross] {len(df)} rows remain after dropping rows with no clear "
              f"current side or missing forward Coinbase data")

    if df["window_id"].nunique() < 20:
        log.warning("Not enough rows with a usable strike-cross label yet. Skipping.")
        return None, {}

    pos_rate = df["label_strike_cross"].mean()
    n_pos = int(df["label_strike_cross"].sum())
    n_neg = len(df) - n_pos
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    log.info(f"[StrikeCross] label balance: {pos_rate*100:.2f}% positive "
              f"(n_pos={n_pos}, n_neg={n_neg}), scale_pos_weight={scale_pos_weight:.2f}")

    aucs, briers = [], []
    for train_df, test_df in walk_forward_splits(df, n_splits=3):
        if train_df.empty or test_df.empty:
            continue
        model = xgb.XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(train_df[X_cols], train_df["label_strike_cross"])
        preds = model.predict_proba(test_df[X_cols])[:, 1]
        if test_df["label_strike_cross"].nunique() > 1:
            aucs.append(roc_auc_score(test_df["label_strike_cross"], preds))
        briers.append(brier_score_loss(test_df["label_strike_cross"], preds))

    log.info(f"[StrikeCross] walk-forward AUC: {np.mean(aucs) if aucs else float('nan'):.4f} | "
              f"Brier: {np.mean(briers):.4f}")

    final_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    final_model.fit(df[X_cols], df["label_strike_cross"])
    return final_model, {
        "auc": float(np.mean(aucs)) if aucs else None,
        "brier": float(np.mean(briers)),
        "pos_rate": float(pos_rate),
        "scale_pos_weight": float(scale_pos_weight),
    }


def save_artifacts(reversal_model, strike_cross_model, metrics: dict, feature_cols: list[str]):
    """NOTE: directional/settlement model intentionally no longer trained
    or saved -- scrapped in favor of focusing on reversal and (now)
    strike-crossing, both of which ask more directly trade-relevant
    questions than "where will price settle"."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    if reversal_model is not None:
        reversal_model.save_model(os.path.join(ARTIFACT_DIR, "flip_model.json"))
    if strike_cross_model is not None:
        strike_cross_model.save_model(os.path.join(ARTIFACT_DIR, "strike_cross_model.json"))
    with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
        json.dump({
            "trained_at": datetime.utcnow().isoformat(),
            "feature_cols": feature_cols,
            "metrics": metrics,
            "note": "flip_model.json holds the general-purpose reversal model (usable at "
                    "any point in the window). strike_cross_model.json predicts whether "
                    "price actually crosses the strike within the next horizon -- the "
                    "settlement/directional model was scrapped entirely.",
        }, f, indent=2)

    try:
        s3 = boto3.client("s3", region_name=REGION)
        for fname in ["flip_model.json", "strike_cross_model.json", "metadata.json"]:
            local_path = os.path.join(ARTIFACT_DIR, fname)
            if os.path.exists(local_path):
                s3.upload_file(local_path, S3_BUCKET, f"models/{fname}")
        log.info(f"Uploaded artifacts to s3://{S3_BUCKET}/models/")
    except Exception as e:
        log.warning(f"S3 upload failed (continuing -- local artifacts still saved): {e}")


def main():
    df = load_dataset()
    reversal_model, reversal_metrics = train_reversal_model(df)
    strike_cross_model, strike_cross_metrics = train_strike_cross_model(df)
    save_artifacts(reversal_model, strike_cross_model,
                    {"reversal": reversal_metrics, "strike_cross": strike_cross_metrics},
                    [c for c in FEATURE_COLS if c in df.columns])
    log.info("Training complete.")


if __name__ == "__main__":
    main()
