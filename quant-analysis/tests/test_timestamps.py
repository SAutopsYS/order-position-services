from datetime import date

import numpy as np
import pandas as pd

from src.timestamps import (
    align_announcement_to_bar,
    first_complete_bar,
    infer_sessions,
    market_timestamps_are_sorted,
    parse_announcement_timestamp,
)


def _bars(*values: str) -> np.ndarray:
    return pd.to_datetime(list(values)).to_numpy(dtype="datetime64[ns]")


def test_valid_dissemdt_is_preferred() -> None:
    result = parse_announcement_timestamp(
        {
            "DissemDT": "2024-01-02T09:15:00",
            "DT_TM": "2024-01-02T10:00:00",
        }
    )
    assert result.source == "DissemDT"
    assert result.timestamp == pd.Timestamp("2024-01-02T09:15:00")
    assert result.error is None


def test_dt_tm_used_when_dissemdt_invalid() -> None:
    result = parse_announcement_timestamp(
        {
            "DissemDT": "",
            "DT_TM": "2024-01-02T10:00:00",
        }
    )
    assert result.source == "DT_TM"
    assert result.timestamp == pd.Timestamp("2024-01-02T10:00:00")


def test_both_timestamps_invalid_are_excluded() -> None:
    result = parse_announcement_timestamp({"DissemDT": "", "DT_TM": "not-a-time"})
    assert result.timestamp is None
    assert result.error == "both DissemDT and DT_TM unusable"


def test_mixed_iso_precision_is_accepted() -> None:
    with_fraction = parse_announcement_timestamp(
        {"DissemDT": "2026-08-19T23:19:39.42", "DT_TM": ""}
    )
    without_fraction = parse_announcement_timestamp(
        {"DissemDT": "2026-07-17T16:03:54", "DT_TM": ""}
    )
    assert with_fraction.timestamp is not None
    assert without_fraction.timestamp is not None


def test_exact_0915_maps_to_0915_bar() -> None:
    bars = _bars("2024-01-02T09:15:00", "2024-01-02T09:16:00")
    chosen = first_complete_bar(pd.Timestamp("2024-01-02T09:15:00"), bars)
    assert chosen == pd.Timestamp("2024-01-02T09:15:00")


def test_intramute_announcement_maps_to_next_minute() -> None:
    bars = _bars("2024-01-02T09:15:00", "2024-01-02T09:16:00")
    chosen = first_complete_bar(pd.Timestamp("2024-01-02T09:15:30"), bars)
    assert chosen == pd.Timestamp("2024-01-02T09:16:00")


def test_premarket_maps_to_first_session_bar() -> None:
    bars = _bars("2024-01-02T09:15:00", "2024-01-02T09:16:00")
    chosen = first_complete_bar(pd.Timestamp("2024-01-02T08:00:00"), bars)
    assert chosen == pd.Timestamp("2024-01-02T09:15:00")


def test_after_market_maps_to_next_session_bar() -> None:
    bars = _bars(
        "2024-01-02T09:15:00",
        "2024-01-02T15:29:00",
        "2024-01-03T09:15:00",
    )
    chosen = first_complete_bar(pd.Timestamp("2024-01-02T16:00:00"), bars)
    assert chosen == pd.Timestamp("2024-01-03T09:15:00")


def test_weekend_maps_to_next_trading_session() -> None:
    bars = _bars("2024-01-05T09:15:00", "2024-01-08T09:15:00")
    chosen = first_complete_bar(pd.Timestamp("2024-01-06T11:00:00"), bars)
    assert chosen == pd.Timestamp("2024-01-08T09:15:00")


def test_no_bar_before_dissemination_is_selected() -> None:
    bars = _bars("2024-01-02T09:15:00", "2024-01-02T09:16:00")
    chosen = first_complete_bar(pd.Timestamp("2024-01-02T09:15:01"), bars)
    assert chosen == pd.Timestamp("2024-01-02T09:16:00")
    assert chosen >= pd.Timestamp("2024-01-02T09:15:01")


def test_alignment_includes_inferred_session() -> None:
    market = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02T09:15:00", "2024-01-02T15:29:00", "2024-01-03T09:15:00"]
            )
        }
    )
    sessions = infer_sessions(market)
    aligned = align_announcement_to_bar(
        pd.Timestamp("2024-01-02T16:00:00"),
        market["timestamp"].to_numpy(dtype="datetime64[ns]"),
        sessions,
    )
    assert aligned.status == "aligned"
    assert aligned.effective_trading_date == date(2024, 1, 3)
    assert aligned.effective_bar_timestamp == pd.Timestamp("2024-01-03T09:15:00")


def test_no_later_bar_is_excluded() -> None:
    bars = _bars("2024-01-02T09:15:00")
    aligned = align_announcement_to_bar(
        pd.Timestamp("2024-01-03T10:00:00"), bars, []
    )
    assert aligned.status == "excluded"
    assert aligned.exclusion_reason == "no complete bar at or after dissemination"
