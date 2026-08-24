from pathlib import Path

import pandas as pd

from src.final_evidence import (
    REQUIRED_EVIDENCE_FIELDS,
    REQUIRED_INSIGHT_FIELDS,
    build_candidate_insights,
    build_evidence_table,
    build_figure_manifest,
    has_surprise_fields,
    unsupported_subjects,
)
from src.subject_analysis import STAGE2_SUBJECTS


def _min_returns() -> pd.DataFrame:
    rows = []
    for subject in STAGE2_SUBJECTS:
        for horizon in ("5m", "session_close", "d5"):
            rows.append(
                {
                    "subject_group": subject,
                    "horizon": horizon,
                    "n": 20 if subject != "corporate_actions" else 8,
                    "mean_return": -0.005 if subject == "financial_results" else 0.001,
                    "median_return": -0.006 if subject == "financial_results" else 0.0,
                    "standard_error": 0.002,
                    "ci95_lower": -0.009,
                    "ci95_upper": -0.001 if subject == "financial_results" else 0.005,
                    "independent_event_count": 20,
                    "analysis_status": "inferential",
                }
            )
    return pd.DataFrame(rows)


def _min_volume() -> pd.DataFrame:
    rows = []
    for subject in STAGE2_SUBJECTS:
        rows.append(
            {
                "subject_group": subject,
                "horizon": "5m",
                "n": 20,
                "mean_volume_ratio": 10.0 if subject == "financial_results" else 1.2,
                "median_volume_ratio": 6.0 if subject == "financial_results" else 1.0,
                "independent_event_count": 20,
            }
        )
    return pd.DataFrame(rows)


def _min_ranges() -> pd.DataFrame:
    rows = []
    for subject in STAGE2_SUBJECTS:
        rows.append(
            {
                "subject_group": subject,
                "horizon": "session_close",
                "n": 20,
                "mean_range": 0.04 if subject == "financial_results" else 0.02,
                "median_range": 0.03 if subject == "financial_results" else 0.015,
                "independent_event_count": 20,
            }
        )
    return pd.DataFrame(rows)


def _min_quality() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_group": "capital_structure",
                "announcement_count": 50,
                "cluster_count": 20,
                "independent_event_count": 20,
                "largest_cluster_size": 5,
                "announcement_to_cluster_ratio": 2.5,
            }
        ]
    )


def _min_pead() -> pd.DataFrame:
    rows = []
    for reaction, mean in (("all", -0.01), ("positive", 0.02), ("negative", -0.02)):
        for horizon in ("d1", "d3", "d5", "d10", "d20"):
            rows.append(
                {
                    "scope": "pooled",
                    "symbol": "ALL",
                    "initial_reaction": reaction,
                    "horizon": horizon,
                    "n": 26 if reaction == "positive" else 49 if reaction == "negative" else 75,
                    "mean_return": mean,
                    "median_return": mean,
                    "ci95_lower": mean - 0.01,
                    "ci95_upper": mean + 0.01,
                    "sample_note": "inferential",
                }
            )
    rows.append(
        {
            "scope": "stock",
            "symbol": "HAL",
            "initial_reaction": "all",
            "horizon": "d1",
            "n": 13,
            "mean_return": 0.01,
            "median_return": 0.01,
            "ci95_lower": -0.01,
            "ci95_upper": 0.03,
            "sample_note": "inferential",
        }
    )
    return pd.DataFrame(rows)


def _min_missing() -> pd.DataFrame:
    return pd.DataFrame(
        [{"subject_group": "regulatory_compliance", "horizon": "60m", "missing_rate": 0.1}]
    )


def test_required_subject_groups_are_preserved() -> None:
    returns = _min_returns()
    assert set(STAGE2_SUBJECTS).issubset(set(returns["subject_group"]))


def test_evidence_and_insights_have_required_fields() -> None:
    insights = build_candidate_insights(
        _min_returns(),
        _min_volume(),
        _min_ranges(),
        _min_quality(),
        _min_pead(),
        _min_missing(),
    )
    evidence = build_evidence_table(insights, _min_returns(), _min_volume(), _min_pead())
    for field in REQUIRED_INSIGHT_FIELDS:
        assert field in insights.columns
    for field in REQUIRED_EVIDENCE_FIELDS:
        assert field in evidence.columns
    assert insights["n"].notna().all()
    assert insights["limitation"].map(lambda text: len(str(text)) > 10).all()


def test_pead_is_price_conditioned() -> None:
    insights = build_candidate_insights(
        _min_returns(),
        _min_volume(),
        _min_ranges(),
        _min_quality(),
        _min_pead(),
        _min_missing(),
    )
    pead = insights[insights["insight_id"] == "I4"].iloc[0]
    blob = " ".join(str(pead[col]) for col in insights.columns)
    assert "price-conditioned" in blob.lower()
    assert "earnings-surprise" in blob.lower()


def test_no_surprise_fields_or_new_subjects() -> None:
    insights = build_candidate_insights(
        _min_returns(),
        _min_volume(),
        _min_ranges(),
        _min_quality(),
        _min_pead(),
        _min_missing(),
    )
    evidence = build_evidence_table(insights, _min_returns(), _min_volume(), _min_pead())
    assert has_surprise_fields(insights) is False
    assert has_surprise_fields(evidence) is False
    assert unsupported_subjects(evidence) == []


def test_figure_manifest_references_existing_files(tmp_path: Path) -> None:
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    names = [
        "fig01_subject_session_close_returns.png",
        "fig02_subject_d5_returns.png",
        "fig03_subject_volume_ratios.png",
        "fig04_subject_session_close_range.png",
        "fig05_pead_price_conditioned.png",
        "fig06_stock_pead_descriptive.png",
        "fig07_subject_event_counts.png",
        "fig08_financial_results_horizons.png",
    ]
    for name in names:
        (figure_dir / name).write_bytes(b"png")
    manifest = build_figure_manifest(figure_dir)
    assert manifest["exists"].all()
    assert set(manifest["filename"]) == set(names)


def test_chart_source_outputs_exist() -> None:
    root = Path("outputs")
    required = [
        root / "subject_impact_summary.csv",
        root / "subject_volume_summary.csv",
        root / "subject_range_summary.csv",
        root / "subject_cluster_quality.csv",
        root / "pead_summary.csv",
        root / "event_level_results.csv",
        root / "event_clusters.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    assert missing == []


def test_no_em_dash_in_project_text() -> None:
    roots = [Path("src"), Path("tests"), Path("README.md")]
    offenders = []
    for root in roots:
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if path.is_file() and path.suffix in {".py", ".md", ".txt", ".csv"}:
                text = path.read_text(encoding="utf-8")
                if "\u2014" in text:
                    offenders.append(str(path))
    assert offenders == []
