"""Announcement timestamp parsing, session lookup, and bar alignment.

Timezone assumption: timestamps are stored as naive datetimes in the form
supplied by the files. DATA_FORMAT.md was not included in the assessment pack,
so no timezone is invented or converted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimestampParse:
    timestamp: Optional[pd.Timestamp]
    source: Optional[str]
    error: Optional[str] = None


@dataclass(frozen=True)
class Session:
    trading_date: date
    start: pd.Timestamp
    end: pd.Timestamp
    index: int


@dataclass(frozen=True)
class BarAlignment:
    effective_bar_timestamp: Optional[pd.Timestamp]
    effective_trading_date: Optional[date]
    session_start: Optional[pd.Timestamp]
    session_end: Optional[pd.Timestamp]
    session_index: Optional[int]
    status: str
    exclusion_reason: Optional[str] = None


def parse_raw_timestamp(value: object) -> Optional[pd.Timestamp]:
    """Parse one timestamp value with mixed ISO precision allowed."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value

    parsed = pd.to_datetime(value, format="ISO8601", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def parse_announcement_timestamp(row: dict) -> TimestampParse:
    """Prefer DissemDT when valid. Fall back to DT_TM. Never invent a value."""
    dissem = parse_raw_timestamp(row.get("DissemDT"))
    if dissem is not None:
        return TimestampParse(timestamp=dissem, source="DissemDT")

    dt_tm = parse_raw_timestamp(row.get("DT_TM"))
    if dt_tm is not None:
        return TimestampParse(timestamp=dt_tm, source="DT_TM")

    return TimestampParse(
        timestamp=None,
        source=None,
        error="both DissemDT and DT_TM unusable",
    )


def market_timestamps_are_sorted(timestamps: pd.Series) -> bool:
    return bool(timestamps.is_monotonic_increasing)


def infer_sessions(market: pd.DataFrame, timestamp_col: str = "timestamp") -> list[Session]:
    """Build sessions from actual bars. Do not assume a weekday or clock calendar."""
    if market.empty:
        return []
    frame = market[[timestamp_col]].copy()
    frame["trading_date"] = frame[timestamp_col].dt.date
    grouped = frame.groupby("trading_date", sort=True)[timestamp_col]
    sessions: list[Session] = []
    for index, (trading_date, series) in enumerate(grouped):
        sessions.append(
            Session(
                trading_date=trading_date,
                start=pd.Timestamp(series.min()),
                end=pd.Timestamp(series.max()),
                index=index,
            )
        )
    return sessions


def session_for_bar(bar_ts: pd.Timestamp, sessions: list[Session]) -> Optional[Session]:
    bar_date = bar_ts.date()
    for session in sessions:
        if session.trading_date == bar_date:
            return session
    return None


def first_complete_bar(
    announcement_ts: pd.Timestamp,
    bar_times: np.ndarray,
) -> Optional[pd.Timestamp]:
    """First one-minute bar whose start is at or after dissemination.

    Uses searchsorted. Does not scan every market row per event.
    A bar that began before the announcement is never selected.
    """
    if bar_times.size == 0:
        return None
    position = int(np.searchsorted(bar_times, np.datetime64(announcement_ts), side="left"))
    if position >= bar_times.size:
        return None
    return pd.Timestamp(bar_times[position])


def align_announcement_to_bar(
    announcement_ts: pd.Timestamp,
    bar_times: np.ndarray,
    sessions: list[Session],
) -> BarAlignment:
    bar_ts = first_complete_bar(announcement_ts, bar_times)
    if bar_ts is None:
        return BarAlignment(
            effective_bar_timestamp=None,
            effective_trading_date=None,
            session_start=None,
            session_end=None,
            session_index=None,
            status="excluded",
            exclusion_reason="no complete bar at or after dissemination",
        )
    session = session_for_bar(bar_ts, sessions)
    return BarAlignment(
        effective_bar_timestamp=bar_ts,
        effective_trading_date=bar_ts.date(),
        session_start=None if session is None else session.start,
        session_end=None if session is None else session.end,
        session_index=None if session is None else session.index,
        status="aligned",
    )
