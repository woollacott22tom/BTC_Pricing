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
    bid_depth_top10: float | None = None  # summed size, top 10 bid levels
    ask_depth_top10: float | None = None  # summed size, top 10 ask levels


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


def spread(tick: Tick) -> float | None:
    if tick.best_bid is None or tick.best_ask is None:
        return None
    return tick.best_ask - tick.best_bid


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


def distance_to_strike_in_stdevs(current_price: float, strike_price: float, vol_1s: float) -> float | None:
    """How many trailing-volatility standard deviations current price is
    away from the window's opening price (the settlement threshold)."""
    if vol_1s <= 0 or strike_price == 0:
        return None
    log_dist = np.log(current_price / strike_price)
    return log_dist / vol_1s


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
    }
    return feat
