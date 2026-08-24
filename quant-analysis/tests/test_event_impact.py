from datetime import datetime, timedelta

import pandas as pd

from src.event_impact import (
    build_market_index,
    compute_event_metrics,
    mean_confidence_interval,
)


def _bars(start: str, minutes: int, close_start: float = 100.0, volume: float = 10.0) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    rows = []
    for offset in range(minutes):
        ts = start_ts + timedelta(minutes=offset)
        close = close_start + offset * 0.1
        rows.append(
            {
                "timestamp": ts,
                "open": close,
                "high": close + 0.05,
                "low": close - 0.05,
                "close": close,
                "volume": volume,
            }
        )
    return pd.DataFrame(rows)


def _event(**overrides) -> dict:
    base = {
        "event_id": "evt-1",
        "symbol": "RELIANCE",
        "cluster_id": "clu_evt-1",
        "cluster_size": 1,
        "subject_group": "credit_ratings",
        "is_financial_result": False,
        "chosen_timestamp": "2024-01-08T09:15:00",
        "effective_bar_timestamp": "2024-01-08T09:15:00",
        "effective_trading_date": "2024-01-08",
        "alignment_status": "aligned",
        "exclusion_reason": "",
    }
    base.update(overrides)
    return base


def _index_from_sessions(sessions: list[pd.DataFrame]) -> object:
    market = pd.concat(sessions, ignore_index=True)
    return build_market_index(market, "RELIANCE")


def test_five_minute_return() -> None:
    prior = [_bars(f"2024-01-0{day}T09:15:00", 20) for day in range(2, 8)]
    day = _bars("2024-01-08T09:15:00", 20)
    index = _index_from_sessions(prior + [day])
    result = compute_event_metrics(_event(), index)
    assert result["usable_5m"] is True
    assert abs(result["return_5m"] - (100.5 / 100.0 - 1)) < 1e-9


def test_intraday_returns_use_timestamp_offsets() -> None:
    prior = [_bars(f"2024-01-0{day}T09:15:00", 70, volume=8.0) for day in range(2, 8)]
    event_day = _bars("2024-01-08T09:15:00", 70, close_start=100.0, volume=20.0)
    index = _index_from_sessions(prior + [event_day])
    result = compute_event_metrics(_event(), index)
    assert abs(result["return_5m"] - (100.5 / 100.0 - 1)) < 1e-9
    assert abs(result["return_30m"] - (103.0 / 100.0 - 1)) < 1e-9
    assert abs(result["return_60m"] - (106.0 / 100.0 - 1)) < 1e-9


def test_session_close_uses_last_actual_bar() -> None:
    prior = [_bars(f"2024-01-0{day}T09:15:00", 10, volume=8.0) for day in range(2, 8)]
    event_day = _bars("2024-01-08T09:15:00", 10, close_start=100.0)
    index = _index_from_sessions(prior + [event_day])
    result = compute_event_metrics(_event(), index)
    last_close = 100.0 + 9 * 0.1
    assert abs(result["return_session_close"] - (last_close / 100.0 - 1)) < 1e-9
    assert result["usable_session_close"] is True


def test_multi_day_horizons() -> None:
    sessions = []
    start = datetime(2024, 1, 2, 9, 15)
    for i in range(26):
        day = start + timedelta(days=i)
        sessions.append(_bars(day.isoformat(), 10, close_start=100.0 + i))
    index = _index_from_sessions(sessions)
    event = _event(
        effective_bar_timestamp="2024-01-02T09:15:00",
        chosen_timestamp="2024-01-02T09:15:00",
        effective_trading_date="2024-01-02",
    )
    result = compute_event_metrics(event, index)
    assert result["usable_d1"] is True
    assert result["usable_d5"] is True
    assert result["usable_d10"] is True
    assert result["usable_d20"] is True
    assert abs(result["return_d1"] - ((101.0 + 0.9) / 100.0 - 1)) < 1e-9
    assert abs(result["return_d5"] - ((105.0 + 0.9) / 100.0 - 1)) < 1e-9
    assert abs(result["return_d10"] - ((110.0 + 0.9) / 100.0 - 1)) < 1e-9
    assert abs(result["return_d20"] - ((120.0 + 0.9) / 100.0 - 1)) < 1e-9


def test_anchor_is_not_a_previous_bar() -> None:
    day = _bars("2024-01-08T09:15:00", 20, close_start=50.0)
    day.loc[0, "close"] = 50.0
    day.loc[1, "close"] = 80.0
    prior = [_bars(f"2024-01-0{d}T09:15:00", 20) for d in range(2, 8)]
    index = _index_from_sessions(prior + [day])
    result = compute_event_metrics(
        _event(effective_bar_timestamp="2024-01-08T09:16:00"),
        index,
    )
    assert result["anchor_close"] == 80.0
    assert pd.Timestamp(result["effective_bar_timestamp"]) >= pd.Timestamp("2024-01-08T09:16:00")


def test_intraday_does_not_cross_session_boundary() -> None:
    prior = [_bars(f"2024-01-0{d}T09:15:00", 6) for d in range(2, 8)]
    short = _bars("2024-01-08T09:15:00", 6)
    index = _index_from_sessions(prior + [short])
    result = compute_event_metrics(_event(), index)
    assert result["usable_5m"] is True
    assert result["usable_30m"] is False
    assert "session_ended_before_horizon" in result["missing_reason"]


def test_weekend_uses_next_trading_session() -> None:
    friday = _bars("2024-01-05T09:15:00", 10, close_start=100.0)
    monday = _bars("2024-01-08T09:15:00", 10, close_start=110.0)
    extras = [_bars(f"2023-12-2{d}T09:15:00", 10) for d in range(5, 10)]
    index = _index_from_sessions(extras + [friday, monday])
    result = compute_event_metrics(
        _event(
            effective_bar_timestamp="2024-01-05T09:15:00",
            chosen_timestamp="2024-01-05T09:15:00",
            effective_trading_date="2024-01-05",
        ),
        index,
    )
    assert result["usable_d1"] is True
    assert abs(result["return_d1"] - ((110.9) / 100.0 - 1)) < 1e-9


def test_missing_future_session_is_explicit() -> None:
    sessions = [_bars(f"2024-01-0{d}T09:15:00", 10) for d in range(2, 8)]
    index = _index_from_sessions(sessions)
    result = compute_event_metrics(
        _event(
            effective_bar_timestamp="2024-01-07T09:15:00",
            chosen_timestamp="2024-01-07T09:15:00",
            effective_trading_date="2024-01-07",
        ),
        index,
    )
    assert result["usable_d20"] is False
    assert "future_session_unavailable" in result["missing_reason"]


def test_invalid_ohlc_marks_metric_unavailable() -> None:
    prior = [_bars(f"2024-01-0{d}T09:15:00", 20) for d in range(2, 8)]
    day = _bars("2024-01-08T09:15:00", 20)
    day.loc[5, "high"] = 1.0
    day.loc[5, "low"] = 2.0
    day.loc[5, "open"] = 3.0
    day.loc[5, "close"] = 3.0
    index = _index_from_sessions(prior + [day])
    result = compute_event_metrics(_event(), index)
    assert result["usable_5m"] is False
    assert "invalid_ohlc" in result["missing_reason"]


def test_volume_baseline_does_not_use_future_data() -> None:
    priors = [_bars(f"2024-01-0{d}T09:15:00", 20, volume=10.0) for d in range(2, 8)]
    event_day = _bars("2024-01-08T09:15:00", 20, volume=40.0)
    future = _bars("2024-01-09T09:15:00", 20, volume=1000.0)
    index = _index_from_sessions(priors + [event_day, future])
    result = compute_event_metrics(_event(), index)
    assert result["volume_ratio_5m"] == 40.0 / 10.0


def test_zero_volume_bar_is_not_automatically_missing() -> None:
    priors = [_bars(f"2024-01-0{d}T09:15:00", 20, volume=10.0) for d in range(2, 8)]
    day = _bars("2024-01-08T09:15:00", 20, volume=0.0)
    index = _index_from_sessions(priors + [day])
    result = compute_event_metrics(_event(), index)
    assert result["usable_5m"] is True
    assert result["return_5m"] == (100.5 / 100.0 - 1)
    assert result["volume_ratio_5m"] == 0.0


def test_cluster_and_event_ids_are_preserved() -> None:
    priors = [_bars(f"2024-01-0{d}T09:15:00", 20) for d in range(2, 8)]
    day = _bars("2024-01-08T09:15:00", 20)
    index = _index_from_sessions(priors + [day])
    result = compute_event_metrics(
        _event(event_id="keep-id", cluster_id="clu_keep", cluster_size=3),
        index,
    )
    assert result["event_id"] == "keep-id"
    assert result["cluster_id"] == "clu_keep"
    assert result["cluster_size"] == 3


def test_confidence_interval_is_deterministic() -> None:
    first = mean_confidence_interval([1.0, 3.0, 5.0])
    again = mean_confidence_interval([1.0, 3.0, 5.0])
    mean, low, high, error = first
    assert error is None
    assert first == again
    assert abs(mean - 3.0) < 1e-9
    empty_mean, _, _, empty_error = mean_confidence_interval([2.0])
    assert empty_mean is None
    assert empty_error == "insufficient_sample"
