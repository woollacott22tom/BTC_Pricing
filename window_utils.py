"""
Utilities for working with Kalshi's fixed 15-minute settlement windows
(boundaries at :00, :15, :30, :45 every hour, UTC).

Settlement mechanics (per Kalshi's stated methodology):
  - Underlying is CF Benchmarks' Real Time Index (RTI).
  - In the final 60 seconds before expiration, 60 one-second RTI prices are
    sampled and averaged. That average is the official settlement value.
  - This means the target variable for training is NOT "price at the exact
    boundary tick" -- it's the mean of the last 60 one-second prints.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta

WINDOW_SECONDS = 15 * 60
SETTLEMENT_AVG_SECONDS = 60


def window_id_for(ts: datetime) -> str:
    """Return the window_id (ISO string of the window's OPEN boundary) that
    a given UTC timestamp falls inside."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts = ts.astimezone(timezone.utc)
    minute_bucket = (ts.minute // 15) * 15
    window_open = ts.replace(minute=minute_bucket, second=0, microsecond=0)
    return window_open.isoformat()


def window_open_close(window_id: str) -> tuple[datetime, datetime]:
    open_ts = datetime.fromisoformat(window_id)
    close_ts = open_ts + timedelta(seconds=WINDOW_SECONDS)
    return open_ts, close_ts


def seconds_remaining(ts: datetime, window_id: str) -> float:
    _, close_ts = window_open_close(window_id)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0.0, (close_ts - ts).total_seconds())


def in_settlement_average_period(ts: datetime, window_id: str) -> bool:
    """True if `ts` falls within the final 60s before window close, i.e. the
    period whose ticks get averaged into the official settlement value."""
    return seconds_remaining(ts, window_id) <= SETTLEMENT_AVG_SECONDS
