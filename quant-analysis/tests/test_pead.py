from datetime import datetime, timedelta

import pandas as pd

from src.event_impact import build_market_index, mean_confidence_interval
from src.pead import (
    build_pead_audit,
    build_pead_event,
    classify_initial_reaction,
    has_surprise_fields,
    select_independent_financial_events,
    summarize_pead,
)


def _bars(start: str, minutes: int = 10, close_start: float = 100.0) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    rows = []
    for offset in range(minutes):
        close = close_start + offset * 0.1
        rows.append(
            {
                "timestamp": start_ts + timedelta(minutes=offset),
                "open": close,
                "high": close + 0.05,
                "low": close - 0.05,
                "close": close,
                "volume": 10.0,
            }
        )
    return pd.DataFrame(rows)


def _cluster(**overrides) -> dict:
    base = {
        "event_id": "evt-1",
        "symbol": "RELIANCE",
        "cluster_id": "clu_a",
        "cluster_size": 1,
        "subject_group": "financial_results",
        "is_financial_result": True,
        "chosen_timestamp": "2024-01-08T09:15:00",
        "effective_bar_timestamp": "2024-01-08T09:15:00",
        "effective_trading_date": "2024-01-08",
        "alignment_status": "aligned",
        "analysis_status": "inferential",
    }
    base.update(overrides)
    return base


def _index(session_count: int = 26, start: str = "2024-01-02T09:15:00") -> object:
    start_ts = datetime.fromisoformat(start)
    sessions = []
    for i in range(session_count):
        day = start_ts + timedelta(days=i)
        sessions.append(_bars(day.isoformat(), close_start=100.0 + i))
    return build_market_index(pd.concat(sessions, ignore_index=True), "RELIANCE")


def test_financial_result_events_are_selected() -> None:
    clusters = pd.DataFrame(
        [
            _cluster(event_id="fr", is_financial_result=True),
            _cluster(event_id="other", is_financial_result=False, cluster_id="clu_b"),
        ]
    )
    selected = select_independent_financial_events(clusters)
    assert list(selected["event_id"]) == ["fr"]


def test_one_observation_per_financial_cluster() -> None:
    clusters = pd.DataFrame(
        [
            _cluster(event_id="first", cluster_id="clu_a", cluster_size=2),
            _cluster(
                event_id="second",
                cluster_id="clu_a",
                cluster_size=2,
                chosen_timestamp="2024-01-08T10:00:00",
                effective_bar_timestamp="2024-01-08T10:00:00",
            ),
        ]
    )
    selected = select_independent_financial_events(clusters)
    assert len(selected) == 1
    assert selected.iloc[0]["cluster_id"] == "clu_a"


def test_earliest_financial_result_is_selected() -> None:
    clusters = pd.DataFrame(
        [
            _cluster(
                event_id="later",
                chosen_timestamp="2024-01-09T09:15:00",
                effective_bar_timestamp="2024-01-09T09:15:00",
                cluster_size=2,
            ),
            _cluster(
                event_id="earlier",
                chosen_timestamp="2024-01-08T09:15:00",
                effective_bar_timestamp="2024-01-08T09:15:00",
                cluster_size=2,
            ),
        ]
    )
    selected = select_independent_financial_events(clusters)
    assert selected.iloc[0]["event_id"] == "earlier"


def test_non_financial_events_are_excluded() -> None:
    clusters = pd.DataFrame(
        [
            _cluster(event_id="board", is_financial_result=False, subject_group="board_governance"),
        ]
    )
    selected = select_independent_financial_events(clusters)
    assert selected.empty


def test_anchor_close_comes_from_effective_bar() -> None:
    index = _index()
    event = _cluster(effective_bar_timestamp="2024-01-02T09:16:00")
    result = build_pead_event(event, index, {"return_session_close": 0.01, "anchor_close": 100.1})
    assert result["anchor_close"] == 100.1
    recomputed = build_pead_event(event, index, {"return_session_close": 0.01})
    assert abs(recomputed["anchor_close"] - 100.1) < 1e-9


def test_pead_horizons_use_trading_sessions() -> None:
    index = _index()
    result = build_pead_event(
        _cluster(effective_bar_timestamp="2024-01-02T09:15:00", effective_trading_date="2024-01-02"),
        index,
        {"return_session_close": 0.02, "anchor_close": 100.0},
    )
    assert result["usable_d1"] is True
    assert result["usable_d3"] is True
    assert result["usable_d5"] is True
    assert result["usable_d10"] is True
    assert result["usable_d20"] is True
    assert abs(result["return_d1"] - ((101.0 + 0.9) / 100.0 - 1)) < 1e-9
    assert abs(result["return_d3"] - ((103.0 + 0.9) / 100.0 - 1)) < 1e-9
    assert abs(result["return_d5"] - ((105.0 + 0.9) / 100.0 - 1)) < 1e-9
    assert abs(result["return_d10"] - ((110.0 + 0.9) / 100.0 - 1)) < 1e-9
    assert abs(result["return_d20"] - ((120.0 + 0.9) / 100.0 - 1)) < 1e-9


def test_calendar_days_are_not_used() -> None:
    friday = _bars("2024-01-05T09:15:00", close_start=100.0)
    monday = _bars("2024-01-08T09:15:00", close_start=110.0)
    tuesday = _bars("2024-01-09T09:15:00", close_start=120.0)
    wednesday = _bars("2024-01-10T09:15:00", close_start=130.0)
    extras = [_bars(f"2024-01-{day:02d}T09:15:00", close_start=140.0 + day) for day in range(11, 31)]
    index = build_market_index(
        pd.concat([friday, monday, tuesday, wednesday, *extras], ignore_index=True),
        "RELIANCE",
    )
    result = build_pead_event(
        _cluster(
            chosen_timestamp="2024-01-05T09:15:00",
            effective_bar_timestamp="2024-01-05T09:15:00",
            effective_trading_date="2024-01-05",
        ),
        index,
        {"return_session_close": -0.01, "anchor_close": 100.0},
    )
    calendar_plus_three = 110.9 / 100.0 - 1
    session_d3 = 130.9 / 100.0 - 1
    assert abs(result["return_d1"] - calendar_plus_three) < 1e-9
    assert abs(result["return_d3"] - session_d3) < 1e-9
    assert abs(result["return_d3"] - calendar_plus_three) > 1e-6


def test_initial_reaction_classes() -> None:
    assert classify_initial_reaction(0.012) == "positive"
    assert classify_initial_reaction(0.0) == "neutral"
    assert classify_initial_reaction(-0.004) == "negative"
    assert classify_initial_reaction(None) == "unavailable"


def test_missing_future_session_is_explicit() -> None:
    index = _index(session_count=4, start="2024-01-02T09:15:00")
    result = build_pead_event(
        _cluster(
            effective_bar_timestamp="2024-01-04T09:15:00",
            effective_trading_date="2024-01-04",
        ),
        index,
        {"return_session_close": 0.01, "anchor_close": 102.0},
    )
    assert result["usable_d20"] is False
    assert pd.isna(result["return_d20"])
    assert "future_session_unavailable" in result["missing_reason"]


def test_summaries_and_uncertainty_are_deterministic() -> None:
    events = pd.DataFrame(
        [
            {
                "event_id": "a",
                "symbol": "HAL",
                "cluster_id": "c1",
                "initial_reaction": "positive",
                "return_d1": 0.02,
                "usable_d1": True,
                "return_d3": 0.03,
                "usable_d3": True,
                "return_d5": 0.04,
                "usable_d5": True,
                "return_d10": 0.05,
                "usable_d10": True,
                "return_d20": 0.06,
                "usable_d20": True,
            },
            {
                "event_id": "b",
                "symbol": "HAL",
                "cluster_id": "c2",
                "initial_reaction": "positive",
                "return_d1": 0.04,
                "usable_d1": True,
                "return_d3": 0.05,
                "usable_d3": True,
                "return_d5": 0.06,
                "usable_d5": True,
                "return_d10": 0.07,
                "usable_d10": True,
                "return_d20": 0.08,
                "usable_d20": True,
            },
            {
                "event_id": "c",
                "symbol": "NYKAA",
                "cluster_id": "c3",
                "initial_reaction": "negative",
                "return_d1": -0.01,
                "usable_d1": True,
                "return_d3": -0.02,
                "usable_d3": True,
                "return_d5": -0.03,
                "usable_d5": True,
                "return_d10": -0.04,
                "usable_d10": True,
                "return_d20": -0.05,
                "usable_d20": True,
            },
        ]
    )
    first = summarize_pead(events)
    second = summarize_pead(events)
    pd.testing.assert_frame_equal(first, second)

    pooled = first[(first["scope"] == "pooled") & (first["initial_reaction"] == "positive") & (first["horizon"] == "d1")].iloc[0]
    assert pooled["n"] == 2
    assert abs(pooled["mean_return"] - 0.03) < 1e-12
    expected_se = pd.Series([0.02, 0.04]).std(ddof=1) / (2 ** 0.5)
    assert abs(pooled["standard_error"] - expected_se) < 1e-12
    mean, low, high, error = mean_confidence_interval([0.02, 0.04])
    assert error is None
    assert abs(pooled["ci95_lower"] - low) < 1e-12
    assert abs(pooled["ci95_upper"] - high) < 1e-12

    stock = first[(first["scope"] == "stock") & (first["symbol"] == "NYKAA") & (first["horizon"] == "d1") & (first["initial_reaction"] == "negative")].iloc[0]
    assert stock["n"] == 1
    assert pd.isna(stock["standard_error"])
    assert stock["sample_note"] == "descriptive"


def test_event_and_cluster_ids_are_preserved() -> None:
    result = build_pead_event(
        _cluster(event_id="keep-id", cluster_id="clu_keep", cluster_size=4),
        _index(),
        {"return_session_close": 0.01, "anchor_close": 100.0},
    )
    assert result["event_id"] == "keep-id"
    assert result["cluster_id"] == "clu_keep"
    assert result["cluster_size"] == 4


def test_no_surprise_based_fields() -> None:
    result = build_pead_event(
        _cluster(),
        _index(),
        {"return_session_close": 0.0, "anchor_close": 100.0},
    )
    frame = pd.DataFrame([result])
    summary = summarize_pead(frame)
    audit = build_pead_audit(pd.DataFrame([_cluster()]), frame)
    assert has_surprise_fields(frame) is False
    assert has_surprise_fields(summary) is False
    assert has_surprise_fields(audit) is False
    assert result["initial_reaction"] == "neutral"
