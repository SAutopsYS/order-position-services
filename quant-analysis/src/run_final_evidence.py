"""Generate Stage 6 charts, evidence table, and candidate insights from existing CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.charts import (
    chart_event_counts,
    chart_financial_horizons,
    chart_pead,
    chart_range,
    chart_stock_pead,
    chart_subject_returns,
    chart_volume_ratios,
)
from src.config import parse_path_args, resolve_paths
from src.final_evidence import (
    build_candidate_insights,
    build_evidence_table,
    build_figure_manifest,
)


def run_final_evidence(data_dir=None, output_dir=None) -> dict:
    if data_dir is None:
        paths = parse_path_args()
    else:
        paths = resolve_paths(data_dir, output_dir or "outputs")
    out = Path(paths.output_dir)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    returns = pd.read_csv(out / "subject_impact_summary.csv")
    volume = pd.read_csv(out / "subject_volume_summary.csv")
    ranges = pd.read_csv(out / "subject_range_summary.csv")
    quality = pd.read_csv(out / "subject_cluster_quality.csv")
    pead = pd.read_csv(out / "pead_summary.csv")
    missing = pd.read_csv(out / "subject_missingness.csv")
    stock = pd.read_csv(out / "stock_subject_summary.csv")

    chart_subject_returns(
        returns,
        "session_close",
        figures / "fig01_subject_session_close_returns.png",
        "Subject-group session-close returns (independent events)",
    )
    chart_subject_returns(
        returns,
        "d5",
        figures / "fig02_subject_d5_returns.png",
        "Subject-group D+5 returns (independent events)",
    )
    chart_volume_ratios(volume, figures / "fig03_subject_volume_ratios.png")
    chart_range(ranges, figures / "fig04_subject_session_close_range.png")
    chart_pead(pead, figures / "fig05_pead_price_conditioned.png")
    chart_stock_pead(pead, figures / "fig06_stock_pead_descriptive.png")
    chart_event_counts(quality, figures / "fig07_subject_event_counts.png")
    chart_financial_horizons(returns, figures / "fig08_financial_results_horizons.png")

    insights = build_candidate_insights(returns, volume, ranges, quality, pead, missing, stock)
    evidence = build_evidence_table(insights, returns, volume, pead)
    manifest = build_figure_manifest(figures)

    insights_path = out / "candidate_insights.csv"
    evidence_path = out / "final_evidence_table.csv"
    manifest_path = out / "figure_manifest.csv"
    insights.to_csv(insights_path, index=False)
    evidence.to_csv(evidence_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    print("Candidate insights", len(insights), flush=True)
    print("Figures", int(manifest["exists"].sum()), "of", len(manifest), flush=True)
    print("Wrote", insights_path, flush=True)
    print("Wrote", evidence_path, flush=True)
    print("Wrote", manifest_path, flush=True)
    return {"insights": insights, "evidence": evidence, "manifest": manifest}


def main(argv: list[str] | None = None) -> int:
    paths = parse_path_args(argv)
    run_final_evidence(paths.data_dir, paths.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
