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

# Bump this whenever label or feature COMPUTATION LOGIC changes, even if
# column NAMES stay the same -- protects the resume cache from silently
# serving stale-logic rows. Confirmed this exact failure mode happened
# twice already with label_reversal (Kalshi-odds -> window-scoped ->
# pure real-price momentum), always under the same column name.
FEATURE_SCHEMA_VERSION = 3
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

# Reversal label config: a pure, continuous real-price momentum/flip
# detector -- NOT tied to Kalshi's odds, the window's strike, the
# settlement period, or the 15-minute window boundary in any way. See
# compute_reversal_labels() for the full multi-bucket shape/magnitude/
# majority definition (REVERSAL_HORIZON_SEC below is no longer used by
# it -- kept only in case something else still references it).
REVERSAL_HORIZON_SEC = 45.0
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


def _rolling_mean_lookup(grid_df: pd.DataFrame, roll_series: pd.Series, offset_seconds: float,
                           tolerance_seconds: float = 10.0) -> pd.Series:
    """Looks up roll_series' value at (row's own timestamp + offset_seconds)
    for every row, via merge_asof. offset_seconds can be negative (backward
    -- safe for a feature) or positive (forward -- only used for LABELS
    here, never a feature, since features must never see the future)."""
    lookup = grid_df[["timestamp"]].copy()
    lookup["target_ts"] = lookup["timestamp"] + offset_seconds
    lookup = lookup.sort_values("target_ts")

    source = pd.DataFrame({
        "timestamp": grid_df["timestamp"].reset_index(drop=True),
        "value": roll_series.reset_index(drop=True),
    }).sort_values("timestamp")

    result = pd.merge_asof(
        lookup.rename(columns={"timestamp": "orig_ts", "target_ts": "timestamp"}),
        source, on="timestamp", direction="backward", tolerance=tolerance_seconds,
    )
    result = result.sort_values("orig_ts").reset_index(drop=True)
    result.index = grid_df.index
    return result["value"]


def compute_reversal_labels(grid_df: pd.DataFrame, bucket_seconds: float = 15.0,
                             min_trend_dollars: float = 1.0,
                             tolerance_seconds: float = 10.0) -> pd.Series:
    """Multi-bucket shape+magnitude+majority reversal detector, using ONLY
    Coinbase's own real price (cb_price) -- no strike, no settlement
    window, no Kalshi anywhere.

    For each row at time t, defines four consecutive 15s BACKWARD bucket
    means (B1 nearest t .. B4 farthest) and four consecutive 15s FORWARD
    bucket means (X1 nearest t .. X4 farthest), all computed from a single
    rolling 15s mean of cb_price plus time-shifted lookups against it --
    the same merge_asof technique already used and validated elsewhere in
    this file, just applied 7 times instead of once.

    ESTABLISHED TREND (backward): the delta spanning the 30-60s-before-t
    range (B3-B4, the earliest/leading transition) must be the LARGEST in
    magnitude among the three consecutive backward deltas, and at least
    $min_trend_dollars -- meaning the real move happened 30-60s before t
    and has been leveling off approaching t (t is a plausible apex/plateau,
    not just an arbitrary mid-trend point). Direction = sign(B3-B4).

    CONFIRMED REVERSAL (forward): the delta spanning the 30-60s-after-t
    range (X4-X3, the latest/trailing transition) must ALSO be the largest
    in magnitude among the three consecutive forward deltas, AND must
    point in the OPPOSITE direction from the established trend -- i.e.
    the reversal is still BUILDING 30-60s out, not snapping back toward
    the original trend (which would mean the apparent "reversal" was just
    a continuation of sideways trading, not a genuine flip).

    MAJORITY CHECK: at least 3 of the 4 forward bucket means (X1..X4) must
    sit on the opposite side of the row's OWN current price (cb_price at
    that exact row, not a bucket mean) from the established trend
    direction.

    A row gets label=1 only if ALL THREE conditions hold together.
      1 = confirmed reversal by all three criteria
      0 = a valid, well-shaped trend existed, but the reversal criteria
          weren't all met (still trending, or shape/majority failed)
      NaN = no valid established trend to begin with (backward shape or
          $1 magnitude requirement not met), or insufficient tick data on
          either side to compute all 8 buckets

    No arbitrary percentage/log-return threshold anywhere -- purely
    dollar-based ($1 minimum on the trend side), per explicit correction
    that a flat percentage threshold was an ungrounded assumption."""
    n = len(grid_df)
    if "cb_price" not in grid_df.columns:
        return pd.Series([np.nan] * n, index=grid_df.index)

    dt_index = pd.to_datetime(grid_df["timestamp"], unit="s")
    roll = grid_df.set_index(dt_index)["cb_price"].rolling(f"{int(bucket_seconds)}s").mean()
    roll = roll.reset_index(drop=True)
    roll.index = grid_df.index

    B1 = roll  # (t-15, t] -- the row's own trailing 15s mean, no lookup needed
    B2 = _rolling_mean_lookup(grid_df, roll, -bucket_seconds, tolerance_seconds)
    B3 = _rolling_mean_lookup(grid_df, roll, -2 * bucket_seconds, tolerance_seconds)
    B4 = _rolling_mean_lookup(grid_df, roll, -3 * bucket_seconds, tolerance_seconds)

    X1 = _rolling_mean_lookup(grid_df, roll, bucket_seconds, tolerance_seconds)
    X2 = _rolling_mean_lookup(grid_df, roll, 2 * bucket_seconds, tolerance_seconds)
    X3 = _rolling_mean_lookup(grid_df, roll, 3 * bucket_seconds, tolerance_seconds)
    X4 = _rolling_mean_lookup(grid_df, roll, 4 * bucket_seconds, tolerance_seconds)

    price_at_t = grid_df["cb_price"].reset_index(drop=True)
    B1, B2, B3, B4 = (s.reset_index(drop=True) for s in (B1, B2, B3, B4))
    X1, X2, X3, X4 = (s.reset_index(drop=True) for s in (X1, X2, X3, X4))

    # Backward: three consecutive chronological-forward deltas
    d_late = B1 - B2    # 0-30s before t
    d_mid = B2 - B3     # 15-45s before t
    d_early = B3 - B4   # 30-60s before t -- must dominate

    have_backward = B1.notna() & B2.notna() & B3.notna() & B4.notna()
    trend_valid = (
        have_backward
        & (d_early.abs() >= d_late.abs())
        & (d_early.abs() >= d_mid.abs())
        & (d_early.abs() >= min_trend_dollars)
    )
    trend_direction = np.sign(d_early)

    # Forward: three consecutive chronological-forward deltas
    e_early = X2 - X1   # 0-30s after t
    e_mid = X3 - X2      # 15-45s after t
    e_late = X4 - X3     # 30-60s after t -- must dominate, opposite trend_direction

    have_forward = X1.notna() & X2.notna() & X3.notna() & X4.notna()
    reversal_shape = (
        have_forward
        & (e_late.abs() >= e_early.abs())
        & (e_late.abs() >= e_mid.abs())
        & (np.sign(e_late) == -trend_direction)
        & (e_late != 0)
    )

    # Majority: >=3 of the 4 forward buckets on the opposite side of the
    # row's own current price from the trend direction
    up_trend = trend_direction > 0
    down_trend = trend_direction < 0

    majority_count = pd.Series(0, index=grid_df.index)
    for X in (X1, X2, X3, X4):
        X = pd.Series(X.to_numpy(), index=grid_df.index)
        is_opposite = np.where(up_trend, X.to_numpy() < price_at_t.to_numpy(),
                       np.where(down_trend, X.to_numpy() > price_at_t.to_numpy(), False))
        majority_count = majority_count + is_opposite.astype(int)

    majority_pass = majority_count >= 3

    label = pd.Series([np.nan] * n)
    valid_rows = trend_valid & have_forward
    confirmed = valid_rows & reversal_shape & majority_pass
    not_confirmed = valid_rows & ~(reversal_shape & majority_pass)

    label[confirmed.to_numpy()] = 1
    label[not_confirmed.to_numpy()] = 0
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

    # Continuous, real-price momentum/flip detector -- usable at ANY point,
    # not tied to the window, strike, or settlement period at all. Answers
    # "is the current real BTC price move about to reverse," independent
    # of which Kalshi contract happens to be open -- an upswing that's
    # ending as a new 15-min window opens is exactly the case this should
    # catch, not something it resets or loses track of.
    merged["label_reversal"] = compute_reversal_labels(merged)

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

    # Stamped so the resume cache can detect STALE LOGIC, not just missing
    # columns -- confirmed this failure mode has happened twice now:
    # label_reversal's underlying computation changed (Kalshi-odds -> a
    # window/strike-scoped version -> pure real-price momentum) without
    # its COLUMN NAME ever changing, meaning a column-presence check alone
    # would silently trust an old, wrongly-computed cached row forever.
    merged.attrs["schema_version"] = FEATURE_SCHEMA_VERSION

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
                stale_version = cached_df.attrs.get("schema_version") != FEATURE_SCHEMA_VERSION
                if missing or stale_version:
                    reason = f"missing {missing}" if missing else f"schema_version mismatch (cached={cached_df.attrs.get('schema_version')}, current={FEATURE_SCHEMA_VERSION})"
                    log.warning(f"Cached file for {wid} predates current schema ({reason}) -- rebuilding it")
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

    # CRITICAL: only read back files corresponding to a window_id in the
    # CURRENT windows list, filtered by the CURRENT schema version -- a
    # bare os.listdir() glob (the previous version of this code) blindly
    # includes ANY .pkl file sitting in cache_dir, including orphaned
    # files left over from an earlier attempt using an older, incompatible
    # label/feature schema. Those files' window_ids may never have been
    # visited by this run's per-window loop above at all (e.g. if an
    # earlier partial run processed a different or overlapping window
    # list), meaning they were NEVER schema-validated -- silently
    # corrupting the final concatenated dataset with stale-logic rows
    # while every individual per-window check looked completely correct.
    # Confirmed as the real cause of a near-total label collapse (0 and
    # 19 usable rows out of 1.15M) despite every smaller, direct test
    # showing healthy ~30%/~99% label validity -- the one thing every
    # healthy test had in common was bypassing this exact line.
    expected_filenames = {_safe_cache_filename(w["window_id"]) for w in windows}
    cached_files = []
    skipped_orphaned = 0
    skipped_stale_at_readback = 0
    for fname in os.listdir(cache_dir):
        if not fname.endswith(".pkl"):
            continue
        if fname not in expected_filenames:
            skipped_orphaned += 1
            continue
        path = os.path.join(cache_dir, fname)
        try:
            check_df = pd.read_pickle(path)
            required_cols = {"label_reversal", "label_strike_cross"}
            missing = required_cols - set(check_df.columns)
            stale_version = check_df.attrs.get("schema_version") != FEATURE_SCHEMA_VERSION
            del check_df
            if missing or stale_version:
                skipped_stale_at_readback += 1
                continue
        except Exception:
            skipped_stale_at_readback += 1
            continue
        cached_files.append(path)

    if skipped_orphaned > 0:
        log.warning(f"Excluded {skipped_orphaned} orphaned cache file(s) not corresponding to "
                     f"any window in this run -- likely leftover from an earlier attempt")
    if skipped_stale_at_readback > 0:
        log.warning(f"Excluded {skipped_stale_at_readback} cache file(s) that failed schema "
                     f"validation at final read-back (missing columns or stale schema_version)")

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


IMBALANCE_TIER_CUTOFFS = (0.6, 0.2)  # heavy at +-0.6, medium at +-0.2, per explicit confirmation


def _imbalance_tier(value: float) -> str | None:
    """Classifies a single book_imbalance reading into one of six tiers.
    Returns None for NaN rather than guessing a tier."""
    if pd.isna(value):
        return None
    heavy, medium = IMBALANCE_TIER_CUTOFFS
    if value >= heavy:
        return "heavy_pos"
    elif value >= medium:
        return "medium_pos"
    elif value >= 0.0:
        return "light_pos"
    elif value >= -medium:
        return "light_neg"
    elif value >= -heavy:
        return "medium_neg"
    else:
        return "heavy_neg"


TIER_NAMES = ["heavy_pos", "medium_pos", "light_pos", "light_neg", "medium_neg", "heavy_neg"]


def compute_block_summary(window_id: str, ticks: list[dict], strike_price: float,
                           sub_bucket_seconds: float = 15.0) -> dict | None:
    """Summarizes ONE 15-minute block (window) into a fixed-width feature
    row: VWAP-style price, volume, a book-imbalance TIER PROFILE (% of 15s
    sub-buckets in each of six tiers, chosen over a plain average since
    book imbalance oscillates fast and symmetrically -- a raw mean would
    wash toward zero and hide persistent one-sided pressure), and
    strike-relative metrics (time/mean price above vs below, cross-count
    split front/back half, max distance either direction).

    Buy/sell volume is intentionally NOT included yet -- it's only been
    captured going forward from tonight, so it would be entirely missing
    for all historical training data. Left out here rather than included
    as an always-NaN column, to keep this function's real, current
    behavior honest about what it actually computes today.

    Returns None if there's not enough tick data to summarize meaningfully
    (fewer than 10 ticks) -- consistent with other functions in this file
    treating "not enough data" as a real, explicit case, not a silent
    zero-filled row."""
    if len(ticks) < 10:
        return None

    df = pd.DataFrame(ticks)
    if "timestamp" not in df.columns or "price" not in df.columns:
        return None
    df = df.sort_values("timestamp").reset_index(drop=True)

    window_start_ts = df["timestamp"].iloc[0]
    volumes = df["volume"].astype(float).fillna(0.0) if "volume" in df.columns else pd.Series([0.0] * len(df))
    prices = df["price"].astype(float)

    total_volume = float(volumes.sum())
    vwap = float((prices * volumes).sum() / total_volume) if total_volume > 0 else float(prices.mean())

    # Book-imbalance tier profile: bucket into 15s sub-buckets, classify
    # each sub-bucket's MEAN imbalance, report % of sub-buckets per tier
    tier_counts = {t: 0 for t in TIER_NAMES}
    valid_bucket_count = 0
    if "book_imbalance" in df.columns:
        df["_sub_bucket"] = ((df["timestamp"] - window_start_ts) // sub_bucket_seconds).astype(int)
        bucket_means = df.groupby("_sub_bucket")["book_imbalance"].apply(lambda s: s.astype(float).mean())
        for val in bucket_means:
            tier = _imbalance_tier(val)
            if tier is not None:
                tier_counts[tier] += 1
                valid_bucket_count += 1

    tier_pct = {
        f"imbalance_pct_{t}": (tier_counts[t] / valid_bucket_count if valid_bucket_count > 0 else np.nan)
        for t in TIER_NAMES
    }

    # Strike-relative metrics
    rel = prices - strike_price
    above_mask = rel > 0
    below_mask = rel < 0

    ts_deltas = df["timestamp"].diff().fillna(0.0)
    time_above = float(ts_deltas[above_mask].sum())
    time_below = float(ts_deltas[below_mask].sum())
    mean_price_above = float(prices[above_mask].mean()) if above_mask.any() else np.nan
    mean_price_below = float(prices[below_mask].mean()) if below_mask.any() else np.nan
    max_dist_above = float(rel[above_mask].max()) if above_mask.any() else 0.0
    max_dist_below = float((-rel[below_mask]).max()) if below_mask.any() else 0.0

    sign = np.sign(rel).replace(0, np.nan).ffill().fillna(0)
    crosses = (sign.diff().abs() > 0).fillna(False)
    window_duration = df["timestamp"].iloc[-1] - window_start_ts
    front_half_mask = (df["timestamp"] - window_start_ts) < (window_duration / 2.0)
    cross_count_front = int(crosses[front_half_mask].sum())
    cross_count_back = int(crosses[~front_half_mask].sum())

    row = {
        "window_id": window_id,
        "vwap": vwap,
        "total_volume": total_volume,
        "time_above_strike": time_above,
        "time_below_strike": time_below,
        "mean_price_above_strike": mean_price_above,
        "mean_price_below_strike": mean_price_below,
        "max_dist_above_strike": max_dist_above,
        "max_dist_below_strike": max_dist_below,
        "cross_count_front_half": cross_count_front,
        "cross_count_back_half": cross_count_back,
    }
    row.update(tier_pct)
    return row
MIN_CONFIRMED_STREAK_LENGTH = 3  # per explicit confirmation


def segment_chunks(directions: list[int], min_streak: int = MIN_CONFIRMED_STREAK_LENGTH) -> list[int | None]:
    """Given block directions (+1/-1) in chronological order, assigns each
    block a chunk_id. A chunk spans from the start of one CONFIRMED (3+)
    same-direction streak up to -- but not including -- the block where
    the NEXT confirmed streak begins. Blocks before the very first
    confirmed streak get chunk_id=None (not enough history yet to belong
    to any chunk) -- explicit, not silently zero-filled.

    Verified against a hand-traced 12-block sequence with three confirmed
    streaks and two intervening single-block breaks before shipping."""
    n = len(directions)
    streaks = []  # (start_idx, length)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and directions[j + 1] == directions[i]:
            j += 1
        streaks.append((i, j - i + 1))
        i = j + 1

    confirmed_starts = [s[0] for s in streaks if s[1] >= min_streak]

    chunk_id: list[int | None] = [None] * n
    for k, start in enumerate(confirmed_starts):
        end = confirmed_starts[k + 1] if k + 1 < len(confirmed_starts) else n
        for idx in range(start, end):
            chunk_id[idx] = k
    return chunk_id
AGG_FIELDS = [
    "total_volume", "time_above_strike", "time_below_strike",
    "max_dist_above_strike", "max_dist_below_strike",
    "cross_count_front_half", "cross_count_back_half",
] + [f"imbalance_pct_{t}" for t in TIER_NAMES]


def build_lookback_features(block_rows: list[dict], directions: list[int],
                             chunk_ids: list[int | None]) -> list[dict]:
    """For each block, computes a FIXED-WIDTH feature row combining:
      own_*        -- the block's own summary (from compute_block_summary)
      chunk_*       -- aggregates over the CURRENT chunk so far (from the
                       chunk's start through and including this block)
      prev_chunk_*  -- aggregates over the ENTIRE previous chunk (a fixed,
                       already-complete set of stats, since that chunk has
                       already ended)

    prev_chunk_* fields are NaN for any block in the first chunk of the
    whole sequence (no previous chunk exists yet) -- explicit, not
    silently zero-filled. Blocks with chunk_id=None (not enough history
    for even a first confirmed streak) are skipped entirely -- there's no
    meaningful chunk context to build for them yet.

    Verified against a hand-traced sequence with a clear volume/direction
    contrast between two chunks before shipping."""
    n = len(block_rows)
    output = []

    for i in range(n):
        cid = chunk_ids[i]
        if cid is None:
            continue

        row = {"window_id": block_rows[i]["window_id"], "direction": directions[i], "chunk_id": cid}
        for k, v in block_rows[i].items():
            if k not in ("window_id",):
                row[f"own_{k}"] = v

        # Current chunk so far: from this chunk's start through index i inclusive
        chunk_indices_so_far = [j for j in range(n) if chunk_ids[j] == cid and j <= i]
        row["chunk_length_so_far"] = len(chunk_indices_so_far)
        row["chunk_direction"] = directions[chunk_indices_so_far[0]]
        for field in AGG_FIELDS:
            vals = [block_rows[j][field] for j in chunk_indices_so_far if field in block_rows[j]]
            row[f"chunk_avg_{field}"] = float(np.nanmean(vals)) if vals else np.nan
        vwaps_so_far = [block_rows[j]["vwap"] for j in chunk_indices_so_far]
        row["chunk_net_price_move"] = vwaps_so_far[-1] - vwaps_so_far[0] if len(vwaps_so_far) > 1 else 0.0

        # Previous chunk: the ENTIRE previous chunk, a fixed/complete set of blocks
        if cid > 0:
            prev_indices = [j for j in range(n) if chunk_ids[j] == cid - 1]
        else:
            prev_indices = []

        if prev_indices:
            row["prev_chunk_length"] = len(prev_indices)
            row["prev_chunk_direction"] = directions[prev_indices[0]]
            for field in AGG_FIELDS:
                vals = [block_rows[j][field] for j in prev_indices if field in block_rows[j]]
                row[f"prev_chunk_avg_{field}"] = float(np.nanmean(vals)) if vals else np.nan
            prev_vwaps = [block_rows[j]["vwap"] for j in prev_indices]
            row["prev_chunk_net_price_move"] = prev_vwaps[-1] - prev_vwaps[0] if len(prev_vwaps) > 1 else 0.0
        else:
            row["prev_chunk_length"] = np.nan
            row["prev_chunk_direction"] = np.nan
            for field in AGG_FIELDS:
                row[f"prev_chunk_avg_{field}"] = np.nan
            row["prev_chunk_net_price_move"] = np.nan

        output.append(row)

    return output
def build_block_dataset() -> pd.DataFrame:
    """Assembles the block-level (one row per 15-minute window) training
    dataset for the continuation/reversal model. Fundamentally different
    granularity from load_dataset() (one row per TICK) -- this fetches
    each window's own raw ticks just to SUMMARIZE them into one row, not
    to keep every tick.

    Pipeline: fetch windows in order (same pre-cutoff exclusion as
    load_dataset) -> summarize each block's own ticks -> segment into
    chunks (confirmed 3+ streaks) -> build lookback features (current
    chunk so far + entire previous chunk) -> compute continuation labels.
    """
    windows = list_closed_windows(region_name=REGION)
    windows = [w for w in windows if w.get("source") != "backfill"]
    windows = [
        w for w in windows
        if datetime.fromisoformat(w["window_id"]).timestamp() + 900.0 > ORDER_BOOK_FIX_CUTOFF_TS
    ]
    windows = sorted(windows, key=lambda w: w["window_id"])
    log.info(f"[Continuation] {len(windows)} usable (post-cutoff) windows to summarize")

    block_rows = []
    directions = []
    for i, w in enumerate(windows):
        ticks = get_generic_window_ticks("btc_ticks", w["window_id"], region=REGION)
        if not ticks:
            continue
        ticks_sorted = sorted(ticks, key=lambda t: float(t["timestamp"]))
        strike_price = float(ticks_sorted[0]["price"])
        summary = compute_block_summary(w["window_id"], ticks_sorted, strike_price)
        if summary is None:
            continue
        block_rows.append(summary)
        directions.append(1 if w["outcome"] == "up" else -1)
        if i % 100 == 0:
            log.info(f"[Continuation]   summarized {i}/{len(windows)} windows")

    log.info(f"[Continuation] {len(block_rows)} windows successfully summarized "
              f"(out of {len(windows)} attempted)")

    chunk_ids = segment_chunks(directions)
    feature_rows = build_lookback_features(block_rows, directions, chunk_ids)
    log.info(f"[Continuation] {len(feature_rows)} blocks fall within a confirmed chunk "
              f"(out of {len(block_rows)} summarized) -- blocks before the first confirmed "
              f"3+ streak are excluded, not enough history yet")

    # Labels computed against the ORIGINAL full direction sequence (by
    # position), then attached to feature_rows by matching window_id --
    # keeps label computation correct regardless of which blocks got
    # dropped for lacking chunk context.
    continuation_labels = compute_continuation_labels(directions)
    label_by_window = {block_rows[i]["window_id"]: continuation_labels[i] for i in range(len(block_rows))}

    df = pd.DataFrame(feature_rows)
    df["label_continuation"] = df["window_id"].map(label_by_window)
    return df


def compute_continuation_labels(directions: list) -> list:
    """label[i] = 1 if directions[i+1] == directions[i] (continuation), 0
    if it flips (reversal), evaluated at EVERY consecutive transition
    regardless of streak length. NaN for the last block (no next block to
    compare against). Verified against two hand-worked example sequences
    before shipping."""
    n = len(directions)
    labels = []
    for i in range(n):
        if i == n - 1:
            labels.append(np.nan)
        else:
            labels.append(1.0 if directions[i + 1] == directions[i] else 0.0)
    return labels


CONTINUATION_FEATURE_COLS = None  # populated dynamically from build_block_dataset()'s
                                    # actual output columns in train_continuation_model(),
                                    # since this feature set's shape (own_*/chunk_*/
                                    # prev_chunk_* fields) is naturally derived rather
                                    # than a fixed list like FEATURE_COLS


def train_continuation_model(df: pd.DataFrame):
    """Block-level (one row per 15-min window) continuation/reversal
    model -- 'given the established trend so far, does the NEXT block
    continue it or flip.' Deliberately NOT restricted to only rows in an
    established trend already: every block-to-block transition is a
    labeled row, so 'it won't reverse' rows carry equal weight to 'it
    will' rows, per explicit correction that focusing only on reversals
    would discount half the real signal."""
    exclude_cols = {"window_id", "direction", "chunk_id", "label_continuation"}
    X_cols = [c for c in df.columns if c not in exclude_cols]
    df = df.dropna(subset=["label_continuation"])
    log.info(f"[Continuation] {len(df)} rows have a usable label "
              f"(excludes only the final block of the sequence, which has no next block)")

    if len(df) < MIN_WINDOWS_REQUIRED:
        log.warning(f"Only {len(df)} labeled blocks available (< {MIN_WINDOWS_REQUIRED} minimum). "
                     "Keep collecting data before training this model. Skipping.")
        return None, {}

    pos_rate = df["label_continuation"].mean()
    log.info(f"[Continuation] label balance: {pos_rate*100:.2f}% continuation "
              f"(n={len(df)})")

    aucs, briers = [], []
    for train_df, test_df in walk_forward_splits(df, n_splits=3):
        if train_df.empty or test_df.empty:
            continue
        model = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        )
        model.fit(train_df[X_cols], train_df["label_continuation"])
        preds = model.predict_proba(test_df[X_cols])[:, 1]
        if test_df["label_continuation"].nunique() > 1:
            aucs.append(roc_auc_score(test_df["label_continuation"], preds))
        briers.append(brier_score_loss(test_df["label_continuation"], preds))

    log.info(f"[Continuation] walk-forward AUC: {np.mean(aucs) if aucs else float('nan'):.4f} | "
              f"Brier: {np.mean(briers):.4f}")

    final_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    )
    final_model.fit(df[X_cols], df["label_continuation"])

    importances = dict(zip(X_cols, final_model.feature_importances_.tolist()))
    sorted_importances = dict(sorted(importances.items(), key=lambda x: -x[1])[:15])
    log.info("[Continuation] Top 15 feature importances:")
    for k, v in sorted_importances.items():
        log.info(f"  {k}: {v:.4f}")

    return final_model, {
        "auc": float(np.mean(aucs)) if aucs else None,
        "brier": float(np.mean(briers)),
        "pos_rate": float(pos_rate),
        "feature_cols": X_cols,
        "top_feature_importances": sorted_importances,
    }


def save_artifacts(reversal_model, strike_cross_model, continuation_model, metrics: dict,
                    feature_cols: list[str], continuation_feature_cols: list[str] | None):
    """NOTE: directional/settlement model intentionally no longer trained
    or saved -- scrapped in favor of reversal, strike-crossing, and (now)
    block-level continuation, all of which ask more directly trade-
    relevant questions than "where will price settle". continuation_model
    uses its OWN dynamically-derived feature set (own_*/chunk_*/
    prev_chunk_* columns), saved separately from feature_cols since it is
    structurally different from the tick-level FEATURE_COLS the other two
    models share."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    if reversal_model is not None:
        reversal_model.save_model(os.path.join(ARTIFACT_DIR, "flip_model.json"))
    if strike_cross_model is not None:
        strike_cross_model.save_model(os.path.join(ARTIFACT_DIR, "strike_cross_model.json"))
    if continuation_model is not None:
        continuation_model.save_model(os.path.join(ARTIFACT_DIR, "continuation_model.json"))
    with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
        json.dump({
            "trained_at": datetime.utcnow().isoformat(),
            "feature_cols": feature_cols,
            "continuation_feature_cols": continuation_feature_cols,
            "metrics": metrics,
            "note": "flip_model.json holds the general-purpose reversal model (usable at "
                    "any point in the window). strike_cross_model.json predicts whether "
                    "price actually crosses the strike within the next horizon. "
                    "continuation_model.json is block-level (one prediction per 15-min "
                    "window, not continuous) -- predicts whether the NEXT block continues "
                    "or reverses the current established trend, using chunk-level lookback "
                    "features. The settlement/directional model was scrapped entirely.",
        }, f, indent=2)

    try:
        s3 = boto3.client("s3", region_name=REGION)
        for fname in ["flip_model.json", "strike_cross_model.json", "continuation_model.json", "metadata.json"]:
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

    log.info("=" * 60)
    log.info("Building block-level continuation dataset (separate pipeline, "
              "one row per 15-min window rather than per tick)...")
    block_df = build_block_dataset()
    continuation_model, continuation_metrics = train_continuation_model(block_df)

    save_artifacts(reversal_model, strike_cross_model, continuation_model,
                    {"reversal": reversal_metrics, "strike_cross": strike_cross_metrics,
                     "continuation": continuation_metrics},
                    [c for c in FEATURE_COLS if c in df.columns],
                    continuation_metrics.get("feature_cols"))
    log.info("Training complete.")


if __name__ == "__main__":
    main()
