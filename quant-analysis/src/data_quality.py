"""Data-quality checks. Invalid rows are recorded, not repaired."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def invalid_ohlc_mask(market: pd.DataFrame) -> pd.Series:
    high_ok = market["high"] >= market[["open", "close", "low"]].max(axis=1)
    low_ok = market["low"] <= market[["open", "close", "high"]].min(axis=1)
    return ~(high_ok & low_ok)


def negative_volume_mask(market: pd.DataFrame) -> pd.Series:
    return market["volume"] < 0


def zero_volume_mask(market: pd.DataFrame) -> pd.Series:
    return market["volume"] == 0


def duplicate_timestamp_mask(market: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.Series:
    return market[timestamp_col].duplicated(keep=False)


def unexpected_minute_gaps(market: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Gaps larger than one minute within the same trading date.

    Cross-session gaps are expected and are not returned.
    """
    if market.empty:
        return market.iloc[0:0].copy()
    frame = market.sort_values(timestamp_col).copy()
    frame["trading_date"] = frame[timestamp_col].dt.date
    frame["prev_ts"] = frame.groupby("trading_date")[timestamp_col].shift(1)
    frame["gap_minutes"] = (frame[timestamp_col] - frame["prev_ts"]).dt.total_seconds() / 60.0
    return frame[frame["gap_minutes"] > 1][
        [timestamp_col, "prev_ts", "gap_minutes", "trading_date"]
    ].reset_index(drop=True)


def summarize_issues(
    dataset: str,
    issue_type: str,
    count: int,
    notes: str = "",
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "issue_type": issue_type,
        "count": int(count),
        "notes": notes,
    }


def format_optional_timestamp(value: Optional[pd.Timestamp]) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat()
