import pandas as pd

from src.event_impact import mean_confidence_interval
from src.subject_analysis import (
    STAGE2_SUBJECTS,
    build_cluster_quality,
    build_evidence_ranking,
    build_missingness,
    build_pead_context,
    build_stock_subject_summary,
    build_subject_metric_summary,
    select_independent_subject_events,
    subject_counts,
    summarize_metric,
    unexpected_subjects,
)
from src.taxonomy import analysis_status


def _row(**overrides) -> dict:
    base = {
        "event_id": "e1",
        "symbol": "HAL",
        "cluster_id": "c1",
        "cluster_size": 1,
        "subject_group": "financial_results",
        "effective_bar_timestamp": "2024-01-02T09:15:00",
        "return_5m": 0.02,
        "return_30m": 0.03,
        "return_60m": 0.04,
        "return_session_close": 0.05,
        "return_d1": 0.06,
        "return_d5": 0.07,
        "return_d10": 0.08,
        "return_d20": 0.09,
        "volume_ratio_5m": 2.0,
        "volume_ratio_30m": 1.8,
        "volume_ratio_60m": 1.5,
        "volume_ratio_session_close": 1.2,
        "range_5m": 0.01,
        "range_30m": 0.02,
        "range_60m": 0.03,
        "range_session_close": 0.04,
        "usable_5m": True,
        "usable_30m": True,
        "usable_60m": True,
        "usable_session_close": True,
        "usable_d1": True,
        "usable_d5": True,
        "usable_d10": True,
        "usable_d20": True,
    }
    base.update(overrides)
    return base


def test_subject_names_are_preserved() -> None:
    events = pd.DataFrame(
        [
            _row(subject_group="financial_results"),
            _row(event_id="e2", cluster_id="c2", subject_group="credit_ratings"),
        ]
    )
    counts = subject_counts(events)
    assert set(counts["subject_group"]) == {"financial_results", "credit_ratings"}
    assert unexpected_subjects(counts["subject_group"]) == []


def test_counts_are_deterministic() -> None:
    events = pd.DataFrame(
        [
            _row(event_id="a", cluster_id="c1", subject_group="board_governance", cluster_size=2),
            _row(
                event_id="b",
                cluster_id="c1",
                subject_group="board_governance",
                cluster_size=2,
                effective_bar_timestamp="2024-01-02T10:00:00",
            ),
            _row(event_id="c", cluster_id="c2", subject_group="board_governance"),
            _row(event_id="d", cluster_id="c3", subject_group="corporate_actions"),
        ]
    )
    first = subject_counts(events)
    second = subject_counts(events)
    pd.testing.assert_frame_equal(first, second)
    board = first[first["subject_group"] == "board_governance"].iloc[0]
    assert board["announcement_count"] == 3
    assert board["cluster_count"] == 2
    assert board["independent_event_count"] == 2


def test_independent_event_is_earliest_in_cluster() -> None:
    events = pd.DataFrame(
        [
            _row(
                event_id="later",
                cluster_id="c1",
                effective_bar_timestamp="2024-01-03T09:15:00",
                return_5m=0.99,
            ),
            _row(
                event_id="earlier",
                cluster_id="c1",
                effective_bar_timestamp="2024-01-02T09:15:00",
                return_5m=0.01,
            ),
        ]
    )
    independent = select_independent_subject_events(events)
    assert len(independent) == 1
    assert independent.iloc[0]["event_id"] == "earlier"


def test_mean_median_se_and_ci() -> None:
    values = pd.Series([0.01, 0.03, 0.05])
    summary = summarize_metric(values, "return")
    assert abs(summary["mean_return"] - 0.03) < 1e-12
    assert abs(summary["median_return"] - 0.03) < 1e-12
    expected_se = values.std(ddof=1) / (3 ** 0.5)
    assert abs(summary["standard_error"] - expected_se) < 1e-12
    _, low, high, error = mean_confidence_interval(values)
    assert error is None
    assert abs(summary["ci95_lower"] - low) < 1e-12
    assert abs(summary["ci95_upper"] - high) < 1e-12


def test_small_group_is_descriptive_only() -> None:
    events = pd.DataFrame([_row(event_id=f"e{i}", cluster_id=f"c{i}") for i in range(3)])
    counts = subject_counts(events)
    assert counts.iloc[0]["analysis_status"] == "descriptive_only"
    assert analysis_status(3) == "descriptive_only"
    assert analysis_status(10) == "inferential"


def test_volume_and_range_aggregate() -> None:
    events = pd.DataFrame(
        [
            _row(event_id="a", cluster_id="c1", volume_ratio_5m=2.0, range_session_close=0.04),
            _row(event_id="b", cluster_id="c2", volume_ratio_5m=4.0, range_session_close=0.08),
        ]
    )
    independent = select_independent_subject_events(events)
    counts = subject_counts(events)
    volume = build_subject_metric_summary(independent, counts, ("5m",), "volume_ratio", "volume_ratio")
    ranges = build_subject_metric_summary(independent, counts, ("session_close",), "range", "range")
    assert abs(volume.iloc[0]["mean_volume_ratio"] - 3.0) < 1e-12
    assert abs(ranges.iloc[0]["mean_range"] - 0.06) < 1e-12


def test_stock_wise_grouping() -> None:
    events = pd.DataFrame(
        [
            _row(event_id="h", symbol="HAL", cluster_id="c1"),
            _row(event_id="r", symbol="RVNL", cluster_id="c2", return_5m=-0.01),
        ]
    )
    stock = build_stock_subject_summary(select_independent_subject_events(events))
    symbols = set(stock["symbol"])
    assert symbols == {"HAL", "RVNL"}
    hal = stock[(stock["symbol"] == "HAL") & (stock["horizon"] == "5m")].iloc[0]
    assert abs(hal["mean_return"] - 0.02) < 1e-12


def test_missingness_and_largest_cluster() -> None:
    events = pd.DataFrame(
        [
            _row(event_id="a", cluster_id="c1", cluster_size=3, usable_d20=False, return_d20=None),
            _row(event_id="b", cluster_id="c1", cluster_size=3, usable_d20=True),
            _row(event_id="c", cluster_id="c2", cluster_size=1),
        ]
    )
    independent = select_independent_subject_events(events)
    missing = build_missingness(independent)
    d20 = missing[missing["horizon"] == "d20"].iloc[0]
    assert d20["total_events"] == 2
    assert d20["usable_events"] == 1
    assert d20["missing_events"] == 1
    assert abs(d20["missing_rate"] - 0.5) < 1e-12
    quality = build_cluster_quality(events, independent)
    assert quality.iloc[0]["largest_cluster_size"] == 3
    assert quality.iloc[0]["median_cluster_size"] == 2.0


def test_evidence_ranking_is_deterministic() -> None:
    events = pd.DataFrame(
        [
            _row(event_id="a", cluster_id="c1", subject_group="financial_results", return_5m=0.10),
            _row(event_id="b", cluster_id="c2", subject_group="credit_ratings", return_5m=0.01),
        ]
    )
    independent = select_independent_subject_events(events)
    counts = subject_counts(events)
    returns = build_subject_metric_summary(independent, counts, ("5m",), "return", "return")
    volume = build_subject_metric_summary(independent, counts, ("5m",), "volume_ratio", "volume_ratio")
    ranges = build_subject_metric_summary(independent, counts, ("5m",), "range", "range")
    first = build_evidence_ranking(returns, volume, ranges)
    second = build_evidence_ranking(returns, volume, ranges)
    pd.testing.assert_frame_equal(first, second)
    top = first[first["rank_family"] == "largest_abs_mean_return_5m"].iloc[0]
    assert top["subject_group"] == "financial_results"
    assert top["rank"] == 1


def test_pead_context_is_financial_results_only() -> None:
    pead = pd.DataFrame(
        [
            {
                "scope": "pooled",
                "symbol": "ALL",
                "initial_reaction": "positive",
                "horizon": "d1",
                "n": 26,
                "mean_return": 0.01,
                "median_return": 0.0,
                "ci95_lower": 0.0,
                "ci95_upper": 0.02,
                "sample_note": "inferential",
            },
            {
                "scope": "stock",
                "symbol": "HAL",
                "initial_reaction": "positive",
                "horizon": "d1",
                "n": 6,
                "mean_return": 0.03,
                "median_return": 0.02,
                "ci95_lower": 0.0,
                "ci95_upper": 0.06,
                "sample_note": "descriptive",
            },
        ]
    )
    context = build_pead_context(pead)
    assert set(context["subject_group"]) == {"financial_results"}
    assert set(context["initial_reaction"]) == {"positive"}
    assert "price_conditioned" in context.iloc[0]["conditioning"]


def test_no_new_subject_is_created() -> None:
    events = pd.DataFrame([_row(subject_group="business_updates")])
    counts = subject_counts(events)
    assert set(counts["subject_group"]).issubset(STAGE2_SUBJECTS)
    assert unexpected_subjects(pd.Series(["made_up_group"])) == ["made_up_group"]
