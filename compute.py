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
