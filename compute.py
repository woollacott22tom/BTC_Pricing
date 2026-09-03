"""
Shared feature computation. Same functions are used at:
  - ingestion time (to log a rolling feature snapshot alongside raw ticks)
  - training time (recomputed from stored raw ticks for consistency)
  - live serving time (scored against the in-memory rolling buffer)

Keeping ONE implementation avoids train/serve skew, which is the #1 way
these systems quietly break.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import pandas as pd  # only needed by the block-level (compute_block_summary
                       # and friends) functions added later in this file --
                       # everything else here is deliberately pandas-free
                       # for low-overhead per-tick live scoring


@dataclass
class Tick:
    ts: float  # unix seconds
    price: float
    volume: float = 0.0
    best_bid: float | None = None
    best_ask: float | None = None
    bid_depth_top10: float | None = None  # summed size, top 10 bid levels (kept for schema compat)
    ask_depth_top10: float | None = None  # summed size, top 10 ask levels
    # Additional depth tiers -- lets features distinguish "wall right at the
    # touch" from "wall sitting 30 levels back", which top-10-only can't.
    bid_depth_5: float | None = None
    ask_depth_5: float | None = None
    bid_depth_20: float | None = None
    ask_depth_20: float | None = None
    bid_depth_50: float | None = None
    ask_depth_50: float | None = None


@dataclass
class RollingBuffer:
    """Holds recent ticks for a single window; feeds feature computation.
    Bounded by max_seconds so memory stays flat regardless of tick rate."""
    max_seconds: float = 900.0
    ticks: deque[Tick] = field(default_factory=deque)

    def add(self, tick: Tick) -> None:
        self.ticks.append(tick)
        cutoff = tick.ts - self.max_seconds
        while self.ticks and self.ticks[0].ts < cutoff:
            self.ticks.popleft()

    def since(self, seconds: float, now_ts: float | None = None) -> list[Tick]:
        now_ts = now_ts if now_ts is not None else (self.ticks[-1].ts if self.ticks else 0)
        cutoff = now_ts - seconds
        return [t for t in self.ticks if t.ts >= cutoff]


def realized_vol(ticks: list[Tick]) -> float:
    """Std dev of log returns across the given tick slice. 0.0 if <2 ticks."""
    if len(ticks) < 2:
        return 0.0
    prices = np.array([t.price for t in ticks], dtype=float)
    prices = prices[prices > 0]
    if len(prices) < 2:
        return 0.0
    log_ret = np.diff(np.log(prices))
    return float(np.std(log_ret))


def momentum(ticks: list[Tick]) -> float:
    """Simple return over the slice: (last - first) / first."""
    if len(ticks) < 2:
        return 0.0
    first, last = ticks[0].price, ticks[-1].price
    if first == 0:
        return 0.0
    return (last - first) / first


def acceleration(buf: RollingBuffer, now_ts: float) -> float:
    """Change in momentum: momentum(last 5s) - momentum(prior 5s)."""
    recent = buf.since(5, now_ts)
    prior_window = [t for t in buf.since(10, now_ts) if t not in recent]
    return momentum(recent) - momentum(prior_window)


def book_imbalance(tick: Tick) -> float | None:
    """(bid_depth - ask_depth) / (bid_depth + ask_depth), range [-1, 1].
    Positive = more resting bid size (buy pressure), negative = ask-heavy."""
    if tick.bid_depth_top10 is None or tick.ask_depth_top10 is None:
        return None
    total = tick.bid_depth_top10 + tick.ask_depth_top10
    if total == 0:
        return 0.0
    return (tick.bid_depth_top10 - tick.ask_depth_top10) / total


def tier_imbalance(tick: Tick, tier: int) -> float | None:
    """Same imbalance formula as book_imbalance, but at a specific depth
    tier (5, 20, or 50). Comparing imbalance ACROSS tiers can reveal
    structure a single-depth view can't: e.g. strong buy pressure right at
    the touch (tier 5) but heavy resting size far below on the ask side
    (tier 50) implies a very different situation than uniform imbalance
    across all tiers."""
    bid = getattr(tick, f"bid_depth_{tier}", None)
    ask = getattr(tick, f"ask_depth_{tier}", None)
    if bid is None or ask is None:
        return None
    total = bid + ask
    if total == 0:
        return 0.0
    return (bid - ask) / total


def spread(tick: Tick) -> float | None:
    if tick.best_bid is None or tick.best_ask is None:
        return None
    return tick.best_ask - tick.best_bid


def spread_change_rate(buf: RollingBuffer, now_ts: float, seconds: float = 10.0) -> float | None:
    """Rate of change of the bid-ask spread over the trailing window.
    Widening spreads are a classic market-microstructure precursor signal
    -- market makers often pull back / widen quotes ahead of anticipated
    volatility, so a widening spread can foreshadow a bigger move before
    the price itself has moved much."""
    recent = buf.since(seconds, now_ts)
    spreads = [spread(t) for t in recent]
    spreads = [s for s in spreads if s is not None]
    if len(spreads) < 2 or spreads[0] == 0:
        return None
    return (spreads[-1] - spreads[0]) / spreads[0]


def spread_zscore(buf: RollingBuffer, now_ts: float, lookback_seconds: float = 60.0) -> float | None:
    """How unusual the CURRENT spread is relative to its own recent
    distribution -- a spread that's 3 standard deviations wider than
    normal is a stronger signal than one that's simply 'wide' in absolute
    terms, which varies naturally with the price level."""
    recent = buf.since(lookback_seconds, now_ts)
    spreads = [spread(t) for t in recent]
    spreads = [s for s in spreads if s is not None]
    if len(spreads) < 5:
        return None
    arr = np.array(spreads, dtype=float)
    std = arr.std()
    if std == 0:
        return 0.0
    return float((arr[-1] - arr.mean()) / std)


def rolling_volume(buf: RollingBuffer, now_ts: float, seconds: float = 60.0) -> float | None:
    """Total traded volume (sum of individual trade sizes, in the
    underlying asset's units) over the trailing window -- distinct from
    every other feature here, which describes book/price STATE rather
    than trading ACTIVITY. Returns 0.0 (not None) if there are ticks in
    the window but all report zero volume, vs None if the window is
    completely empty of ticks."""
    recent = buf.since(seconds, now_ts)
    if not recent:
        return None
    return sum(t.volume for t in recent if t.volume is not None)


def depth_thinning_rate(buf: RollingBuffer, now_ts: float, seconds: float = 10.0) -> float | None:
    """Rate of change of total top-10 depth over the trailing window.
    Negative = depth pulling away (thinning) -> elevated flip risk."""
    recent = buf.since(seconds, now_ts)
    depths = [
        (t.bid_depth_top10 + t.ask_depth_top10)
        for t in recent
        if t.bid_depth_top10 is not None and t.ask_depth_top10 is not None
    ]
    if len(depths) < 2:
        return None
    return (depths[-1] - depths[0]) / max(depths[0], 1e-9)


def distance_to_round_number(price: float, round_to: float) -> float:
    """Signed distance from price to the nearest round-number level (e.g.
    round_to=100 for hundred-dollar levels, round_to=1000 for thousand-
    dollar levels). Round numbers act as informal psychological
    support/resistance -- stop-losses, take-profits, and algorithmic
    orders often cluster there. Negative = price is below the nearest
    round level (approaching it from above would mean price is falling
    toward support); positive = above it."""
    nearest = round(price / round_to) * round_to
    return price - nearest


def distance_to_strike_in_stdevs(current_price: float, strike_price: float, vol_1s: float) -> float | None:
    """How many trailing-volatility standard deviations current price is
    away from the window's opening price (the settlement threshold)."""
    if vol_1s <= 0 or strike_price == 0:
        return None
    log_dist = np.log(current_price / strike_price)
    return log_dist / vol_1s


def _extrema_count(values: np.ndarray) -> int | None:
    """Number of local peaks/troughs (direction changes) in a numeric
    series. Generic core shared by both price-shape and spread-shape
    features -- a rough, unbiased stand-in for 'choppiness' or pattern
    complexity, without trying to name a specific chart pattern
    (head-and-shoulders, bull flag, etc.), which tend to be too subjective
    to encode reliably. Choppy/complex paths have more; smooth trends have
    fewer."""
    if len(values) < 3:
        return None
    diffs = np.diff(values)
    signs = np.sign(diffs)
    signs = signs[signs != 0]  # ignore flat/no-change steps
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def _curvature(values: np.ndarray) -> float | None:
    """Mean absolute second difference, normalized by the series' first
    value. High curvature = the series is bending/whipsawing a lot; low
    curvature = smooth, straight-line-like movement."""
    if len(values) < 3 or values[0] == 0:
        return None
    second_diffs = np.diff(values, n=2)
    return float(np.mean(np.abs(second_diffs)) / values[0])


def _ma_cross_count(values: np.ndarray, ma_window: int) -> int | None:
    """Number of times the series crosses its own trailing moving average
    -- a simple proxy for 'oscillating around a level' (more crosses) vs
    'trending away from it' (fewer crosses)."""
    if len(values) < ma_window + 2:
        return None
    ma = np.convolve(values, np.ones(ma_window) / ma_window, mode="valid")
    aligned = values[ma_window - 1:]
    above = aligned > ma
    if len(above) < 2:
        return 0
    return int(np.sum(above[1:] != above[:-1]))


def local_extrema_count(ticks: list[Tick]) -> int | None:
    """Price-path version -- see _extrema_count for the generic core."""
    if len(ticks) < 3:
        return None
    return _extrema_count(np.array([t.price for t in ticks], dtype=float))


def path_curvature(ticks: list[Tick]) -> float | None:
    """Price-path version -- see _curvature for the generic core."""
    if len(ticks) < 3:
        return None
    return _curvature(np.array([t.price for t in ticks], dtype=float))


def ma_cross_count(ticks: list[Tick], ma_window: int = 5) -> int | None:
    """Price-path version -- see _ma_cross_count for the generic core."""
    if len(ticks) < ma_window + 2:
        return None
    return _ma_cross_count(np.array([t.price for t in ticks], dtype=float), ma_window)


def spread_local_extrema_count(ticks: list[Tick]) -> int | None:
    """Same pattern-shape idea as local_extrema_count, applied to the
    bid-ask SPREAD's own path instead of price. A choppy, oscillating
    spread (market makers repeatedly widening/narrowing) looks different
    from a spread that's smoothly and steadily widening or narrowing --
    this distinguishes those cases without naming a specific pattern."""
    spreads = [spread(t) for t in ticks]
    spreads = [s for s in spreads if s is not None]
    if len(spreads) < 3:
        return None
    return _extrema_count(np.array(spreads, dtype=float))


def spread_curvature(ticks: list[Tick]) -> float | None:
    """Curvature of the spread's own path -- see path_curvature for the
    price-path analog."""
    spreads = [spread(t) for t in ticks]
    spreads = [s for s in spreads if s is not None]
    if len(spreads) < 3:
        return None
    return _curvature(np.array(spreads, dtype=float))


def spread_ma_cross_count(ticks: list[Tick], ma_window: int = 5) -> int | None:
    """How often the spread crosses its own trailing moving average --
    frequent crossing suggests the spread is oscillating/noisy rather than
    trending steadily wider or narrower."""
    spreads = [spread(t) for t in ticks]
    spreads = [s for s in spreads if s is not None]
    if len(spreads) < ma_window + 2:
        return None
    return _ma_cross_count(np.array(spreads, dtype=float), ma_window)


def recent_range_ratio(buf: RollingBuffer, now_ts: float, recent_seconds: float = 15.0, prior_seconds: float = 30.0) -> float | None:
    """Ratio of (high-low range in the most recent window) to (high-low
    range in the prior window before that). >1 means volatility/range is
    expanding right now relative to just before; <1 means contracting."""
    recent = buf.since(recent_seconds, now_ts)
    prior_cutoff_ticks = buf.since(prior_seconds, now_ts)
    prior = [t for t in prior_cutoff_ticks if t not in recent]

    if len(recent) < 2 or len(prior) < 2:
        return None

    recent_prices = [t.price for t in recent]
    prior_prices = [t.price for t in prior]
    recent_range = max(recent_prices) - min(recent_prices)
    prior_range = max(prior_prices) - min(prior_prices)

    if prior_range == 0:
        return None
    return recent_range / prior_range


def compute_feature_snapshot(buf: RollingBuffer, strike_price: float, now_ts: float | None = None) -> dict:
    """Full feature vector for the CURRENT moment, used identically at
    ingestion (logging), training (replay), and live serving."""
    if not buf.ticks:
        return {}
    now_ts = now_ts if now_ts is not None else buf.ticks[-1].ts
    last = buf.ticks[-1]

    win_1s = buf.since(1, now_ts)
    win_5s = buf.since(5, now_ts)
    win_15s = buf.since(15, now_ts)
    win_60s = buf.since(60, now_ts)

    vol_5s = realized_vol(win_5s)
    vol_15s = realized_vol(win_15s)
    vol_60s = realized_vol(win_60s)

    feat = {
        "price": last.price,
        "momentum_5s": momentum(win_5s),
        "momentum_15s": momentum(win_15s),
        "momentum_60s": momentum(win_60s),
        "realized_vol_5s": vol_5s,
        "realized_vol_15s": vol_15s,
        "realized_vol_60s": vol_60s,
        "acceleration": acceleration(buf, now_ts),
        "book_imbalance": book_imbalance(last),
        "spread": spread(last),
        "depth_thinning_10s": depth_thinning_rate(buf, now_ts, 10.0),
        "distance_to_strike_stdevs": distance_to_strike_in_stdevs(last.price, strike_price, vol_15s),
        "log_return_from_strike": float(np.log(last.price / strike_price)) if strike_price else None,
        "local_extrema_60s": local_extrema_count(win_60s),
        "path_curvature_60s": path_curvature(win_60s),
        "ma_cross_count_60s": ma_cross_count(win_60s, ma_window=5),
        "recent_range_ratio": recent_range_ratio(buf, now_ts, 15.0, 30.0),
        "imbalance_5": tier_imbalance(last, 5),
        "imbalance_20": tier_imbalance(last, 20),
        "imbalance_50": tier_imbalance(last, 50),
        "spread_change_rate_10s": spread_change_rate(buf, now_ts, 10.0),
        "spread_zscore_60s": spread_zscore(buf, now_ts, 60.0),
        "spread_local_extrema_60s": spread_local_extrema_count(win_60s),
        "spread_curvature_60s": spread_curvature(win_60s),
        "spread_ma_cross_60s": spread_ma_cross_count(win_60s, ma_window=5),
        "distance_to_round_100": distance_to_round_number(last.price, 100.0),
        "distance_to_round_1000": distance_to_round_number(last.price, 1000.0),
        "volume_15s": rolling_volume(buf, now_ts, 15.0),
        "volume_60s": rolling_volume(buf, now_ts, 60.0),
    }
    return feat


def compute_mean_surge_indicator(buf: RollingBuffer, strike_price: float, window_start_ts: float,
                                   now_ts: float, late_window_seconds: float = 120.0) -> dict | None:
    """Live version of the indicator validated in mean_surge_outcome_analysis.py:
      - mean position (early-window average vs strike) AGREES with the late-
        window move -> ~85% directional accuracy in that direction ("strong")
      - mean and late move DISAGREE -> still ~68-71% accuracy in the MEAN's
        direction (the late move never actually flips the majority outcome
        in backtest data, only partially offsets it) -> "weak", not a flip

    Returns None during the first late_window_seconds of a window (no valid
    early/late split exists yet), or if there isn't enough tick data on
    either side of the split to compute a meaningful average.
    """
    if now_ts - window_start_ts < late_window_seconds:
        return None

    late_boundary_ts = now_ts - late_window_seconds
    all_ticks = buf.since(now_ts - window_start_ts, now_ts)
    early_ticks = [t for t in all_ticks if t.ts < late_boundary_ts]
    late_ticks = [t for t in all_ticks if t.ts >= late_boundary_ts]

    if len(early_ticks) < 3 or len(late_ticks) < 2:
        return None

    mean_price = sum(t.price for t in early_ticks) / len(early_ticks)
    mean_position = "above" if mean_price > strike_price else "below"
    late_change = late_ticks[-1].price - early_ticks[-1].price

    if mean_position == "above":
        signal = "strong_up" if late_change > 0 else "weak_up"
    else:
        signal = "strong_down" if late_change < 0 else "weak_down"

    return {
        "mean_position": mean_position,
        "mean_price": mean_price,
        "late_change": late_change,
        "signal": signal,
    }
def build_live_feature_row(cb_feats: dict, kr_feats: dict | None, cc_feats: dict | None,
                             hour_of_day_utc: int, day_of_week: int,
                             kalshi_momentum: dict | None = None) -> dict:
    """Builds ONE row of features matching train.py's exact column naming,
    for live scoring -- the single-row equivalent of build_window_dataframe()
    + add_consensus_features(). Exists specifically to prevent train/serve
    skew (see compute.py's module docstring) -- confirmed this skew already
    happened once: /live was still using pre-multi-exchange-rebuild column
    names (unprefixed, e.g. "momentum_5s" instead of "cb_momentum_5s"),
    silently producing an all-NaN row for every trained model -- loaded
    successfully, scored nothing, no error surfaced anywhere.

    NOTE: does NOT yet include price_diff_* (cross-exchange rolling gap
    deviation) or kalshi_momentum_* columns unless explicitly passed in via
    kalshi_momentum -- those need genuinely new rolling-history state (a
    price-differential buffer, a Kalshi price history) that doesn't exist
    in app.py yet. Those columns will simply be absent from the returned
    dict; the caller should fill them with NaN rather than block scoring
    entirely -- XGBoost handles missing values natively (it learns a
    default split direction for them during training), so a partial row
    still produces a valid, if less complete, prediction rather than none
    at all. train.py's own docstring already flags this as a valid
    approach, just not yet implemented.
    """
    row: dict = {}
    exchange_feats = {"cb": cb_feats, "kr": kr_feats, "cc": cc_feats}

    for prefix, feats in exchange_feats.items():
        if not feats:
            continue
        for feat_name, value in feats.items():
            if feat_name == "seconds_remaining":
                continue  # not a model feature -- handled separately by the caller
            row[f"{prefix}_{feat_name}"] = value

    # Cross-exchange consensus -- must match train.py's add_consensus_features()
    # EXACTLY (same metrics, same n_positive/n_negative/all_agree/mean/dispersion
    # definitions), just computed from three scalar values instead of a
    # dataframe column. Verified against train.py's real pandas output for a
    # known mixed-sign test case before shipping.
    for metric in ["book_imbalance", "imbalance_5", "imbalance_20", "imbalance_50"]:
        vals = []
        for feats in exchange_feats.values():
            if feats is not None and feats.get(metric) is not None:
                vals.append(feats[metric])

        n_available = len(vals)
        if n_available == 0:
            continue

        n_positive = sum(1 for v in vals if v > 0)
        n_negative = sum(1 for v in vals if v < 0)
        row[f"consensus_{metric}_n_positive"] = float(n_positive)
        row[f"consensus_{metric}_n_negative"] = float(n_negative)

        if n_available >= 2:
            row[f"consensus_{metric}_all_agree"] = (
                1.0 if (n_positive == n_available or n_negative == n_available) else 0.0
            )
        # else: leave absent (NaN) -- matches train.py's np.where(n_available >= 2, ..., np.nan)

        mean = sum(vals) / n_available
        row[f"consensus_{metric}_mean"] = mean

        if n_available >= 2:
            # sample variance (ddof=1), matching pandas' default .std() behavior
            variance = sum((v - mean) ** 2 for v in vals) / (n_available - 1)
            row[f"consensus_{metric}_dispersion"] = variance ** 0.5
        # else: leave absent (NaN) -- matches pandas' .std() on a single value

    row["hour_of_day_utc"] = float(hour_of_day_utc)
    row["day_of_week"] = float(day_of_week)

    if kalshi_momentum:
        for k, v in kalshi_momentum.items():
            if v is not None:
                row[k] = v

    return row
@dataclass
class RollingSeries:
    """Generic bounded rolling history of (timestamp, value) pairs --
    used for scalar state tracked over time that isn't a full Tick
    (cross-exchange price gaps, Kalshi's own price). Bounded by
    max_seconds so memory stays flat, same principle as RollingBuffer."""
    max_seconds: float = 900.0
    points: deque = field(default_factory=deque)

    def add(self, ts: float, value: float) -> None:
        self.points.append((ts, value))
        cutoff = ts - self.max_seconds
        while self.points and self.points[0][0] < cutoff:
            self.points.popleft()

    def _since_inclusive(self, seconds: float, now_ts: float) -> list[tuple[float, float]]:
        """Both-ends-inclusive window: [now-seconds, now]. Matches
        train.py's Kalshi momentum boundary (a plain boolean filter, NOT
        .rolling()) -- verified empirically against real pandas output:
        momentum_15s at t=20 over [5,50.5],[10,51.0],[15,50.8],[20,51.5]
        gives 51.5-50.5=1.0, matching exactly."""
        cutoff = now_ts - seconds
        return [(t, v) for t, v in self.points if cutoff <= t <= now_ts]

    def _since_exclusive_start(self, seconds: float, now_ts: float) -> list[tuple[float, float]]:
        """Exclusive-start window: (now-seconds, now]. Matches pandas'
        .rolling(f'{window}s') time-based convention used for the
        price-diff deviation baseline -- verified empirically against
        real pandas .rolling('30s', min_periods=3).mean() output (an
        inclusive-start assumption gives the WRONG answer here; pandas'
        actual window at t=70 over a 30s span is (40,70], not [40,70])."""
        cutoff = now_ts - seconds
        return [(t, v) for t, v in self.points if t > cutoff]

    def rolling_mean(self, seconds: float, now_ts: float, min_periods: int = 3) -> float | None:
        vals = [v for t, v in self._since_exclusive_start(seconds, now_ts)]
        if len(vals) < min_periods:
            return None
        return sum(vals) / len(vals)

    def momentum(self, seconds: float, now_ts: float) -> float | None:
        pts = self._since_inclusive(seconds, now_ts)
        if len(pts) < 2:
            return None
        return pts[-1][1] - pts[0][1]


def compute_price_diff_deviation_features(diff_series_dict: dict, cb_price: float | None,
                                            kr_price: float | None, cc_price: float | None,
                                            now_ts: float,
                                            baseline_secs: tuple = (30.0, 60.0, 120.0)) -> dict:
    """Live equivalent of train.py's add_price_differential_features().
    Updates each pair's rolling gap history with the CURRENT gap, then
    computes deviation-from-rolling-baseline. diff_series_dict must be
    owned by the CALLER as persistent state (e.g. module-level in app.py)
    across repeated calls -- this is what lets the rolling history
    actually accumulate over time rather than resetting every call."""
    row = {}
    pairs = {"cc_cb": (cc_price, cb_price), "kr_cb": (kr_price, cb_price)}
    for pair_name, (other_price, base_price) in pairs.items():
        if other_price is None or base_price is None:
            continue
        gap = other_price - base_price
        series = diff_series_dict[pair_name]
        series.add(now_ts, gap)

        other, base = pair_name.split("_")
        diff_col = f"price_diff_{other}_{base}"
        row[diff_col] = gap
        for window in baseline_secs:
            rolling_mean = series.rolling_mean(window, now_ts, min_periods=3)
            if rolling_mean is not None:
                row[f"{diff_col}_deviation_{int(window)}s"] = gap - rolling_mean
    return row


def compute_kalshi_momentum_features(kalshi_series: RollingSeries, now_ts: float,
                                       lookbacks: dict | None = None) -> dict:
    """Live equivalent of train.py's _kalshi_ticks_to_momentum_df().
    kalshi_series must be owned by the CALLER as persistent state, updated
    every time a new Kalshi price is observed (e.g. in kalshi_poll_loop)."""
    if lookbacks is None:
        lookbacks = {"kalshi_momentum_15s": 15.0, "kalshi_momentum_60s": 60.0}
    row = {}
    for col_name, lookback in lookbacks.items():
        m = kalshi_series.momentum(lookback, now_ts)
        if m is not None:
            row[col_name] = m
    return row
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


def compute_block_summary(window_id: str, ticks: list, strike_price: float,
                           sub_bucket_seconds: float = 15.0) -> dict | None:
    """Summarizes ONE 15-minute block (window) into a fixed-width feature
    row: VWAP-style price, volume, a book-imbalance TIER PROFILE (% of 15s
    sub-buckets in each of six tiers, chosen over a plain average since
    book imbalance oscillates fast and symmetrically -- a raw mean would
    wash toward zero and hide persistent one-sided pressure), and
    strike-relative metrics (time/mean price above vs below, cross-count
    split front/back half, max distance either direction).

    Shared between train.py (historical windows, one full pass) and
    app.py (live serving, called on the in-progress window's own partial
    ticks) -- moved here specifically so both use the IDENTICAL
    computation rather than a hand-copied, driftable duplicate.

    Buy/sell volume is intentionally NOT included yet -- it's only been
    captured going forward from when that ingestion change shipped, so it
    would be entirely missing for all historical training data.

    Returns None if there's not enough tick data to summarize meaningfully
    (fewer than 10 ticks) -- "not enough data" is a real, explicit case,
    not a silent zero-filled row."""
    if len(ticks) < 10:
        return None

    df = pd.DataFrame(ticks)
    if "timestamp" not in df.columns or "price" not in df.columns:
        return None

    # DynamoDB returns all numeric values as decimal.Decimal, not float --
    # cast EVERY numeric column used here up front, right after building
    # the DataFrame, rather than chasing individual fields one at a time
    # as each one happens to hit an operation Decimal doesn't support.
    for col in ("timestamp", "price", "volume", "book_imbalance"):
        if col in df.columns:
            df[col] = df[col].astype(float)

    df = df.sort_values("timestamp").reset_index(drop=True)

    window_start_ts = df["timestamp"].iloc[0]
    volumes = df["volume"].fillna(0.0) if "volume" in df.columns else pd.Series([0.0] * len(df))
    prices = df["price"].astype(float)

    total_volume = float(volumes.sum())
    vwap = float((prices * volumes).sum() / total_volume) if total_volume > 0 else float(prices.mean())

    # Book-imbalance tier profile: bucket into 15s sub-buckets, classify
    # each sub-bucket's MEAN imbalance, report % of sub-buckets per tier
    tier_counts = {t: 0 for t in TIER_NAMES}
    valid_bucket_count = 0
    if "book_imbalance" in df.columns:
        df["_sub_bucket"] = ((df["timestamp"] - window_start_ts) // sub_bucket_seconds).astype(int)
        bucket_means = df.groupby("_sub_bucket")["book_imbalance"].mean()
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


def segment_chunks(directions: list, min_streak: int = MIN_CONFIRMED_STREAK_LENGTH) -> list:
    """Given block directions (+1/-1) in chronological order, assigns each
    block a chunk_id. A chunk spans from the start of one CONFIRMED (3+)
    same-direction streak up to -- but not including -- the block where
    the NEXT confirmed streak begins. Blocks before the very first
    confirmed streak get chunk_id=None."""
    n = len(directions)
    streaks = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and directions[j + 1] == directions[i]:
            j += 1
        streaks.append((i, j - i + 1))
        i = j + 1

    confirmed_starts = [s[0] for s in streaks if s[1] >= min_streak]

    chunk_id = [None] * n
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


def build_lookback_features(block_rows: list, directions: list, chunk_ids: list) -> list:
    """For each block, computes a FIXED-WIDTH feature row combining:
      own_*        -- the block's own summary (from compute_block_summary)
      chunk_*       -- aggregates over the CURRENT chunk so far
      prev_chunk_*  -- aggregates over the ENTIRE previous chunk

    Blocks with chunk_id=None are skipped entirely."""
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

        chunk_indices_so_far = [j for j in range(n) if chunk_ids[j] == cid and j <= i]
        row["chunk_length_so_far"] = len(chunk_indices_so_far)
        row["chunk_direction"] = directions[chunk_indices_so_far[0]]
        for field in AGG_FIELDS:
            vals = [block_rows[j][field] for j in chunk_indices_so_far if field in block_rows[j]]
            row[f"chunk_avg_{field}"] = float(np.nanmean(vals)) if vals else np.nan
        vwaps_so_far = [block_rows[j]["vwap"] for j in chunk_indices_so_far]
        row["chunk_net_price_move"] = vwaps_so_far[-1] - vwaps_so_far[0] if len(vwaps_so_far) > 1 else 0.0

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


def compute_continuation_labels(directions: list) -> list:
    """label[i] = 1 if directions[i+1] == directions[i] (continuation), 0
    if it flips (reversal). NaN for the last block."""
    n = len(directions)
    labels = []
    for i in range(n):
        if i == n - 1:
            labels.append(np.nan)
        else:
            labels.append(1.0 if directions[i + 1] == directions[i] else 0.0)
    return labels
