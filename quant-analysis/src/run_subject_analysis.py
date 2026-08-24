"""Aggregate Stage 3 event metrics by Stage 2 subject groups."""

from __future__ import annotations

import sys

import pandas as pd

from src.config import parse_path_args, resolve_paths
from src.subject_analysis import (
    RANGE_HORIZONS,
    RETURN_HORIZONS,
    VOLUME_HORIZONS,
    build_cluster_quality,
    build_evidence_ranking,
    build_missingness,
    build_pead_context,
    build_stock_subject_summary,
    build_subject_metric_summary,
    build_subject_review,
    select_independent_subject_events,
    subject_counts,
    unexpected_subjects,
)


def run_subject_analysis(data_dir=None, output_dir=None) -> dict:
    if data_dir is None:
        paths = parse_path_args()
    else:
        paths = resolve_paths(data_dir, output_dir or "outputs")
    out = paths.output_dir
    events_path = out / "event_level_results.csv"
    clusters_path = out / "event_clusters.csv"
    if not events_path.is_file():
        raise FileNotFoundError(f"Missing {events_path}. Run Stage 3 before Stage 5.")
    if not clusters_path.is_file():
        raise FileNotFoundError(f"Missing {clusters_path}. Run Stage 2 before Stage 5.")

    events = pd.read_csv(events_path)
    clusters = pd.read_csv(clusters_path)
    extra = clusters[["event_id", "original_subject"]].drop_duplicates("event_id")
    events = events.merge(extra, on="event_id", how="left")

    unknown = unexpected_subjects(events["subject_group"])
    if unknown:
        raise ValueError(f"Unexpected subject groups: {unknown}")

    counts = subject_counts(events)
    independent = select_independent_subject_events(events)
    returns = build_subject_metric_summary(independent, counts, RETURN_HORIZONS, "return", "return")
    volume = build_subject_metric_summary(independent, counts, VOLUME_HORIZONS, "volume_ratio", "volume_ratio")
    ranges = build_subject_metric_summary(independent, counts, RANGE_HORIZONS, "range", "range")
    stock = build_stock_subject_summary(independent)
    missing = build_missingness(independent)
    quality = build_cluster_quality(events, independent)
    ranking = build_evidence_ranking(returns, volume, ranges)

    pead_path = out / "pead_summary.csv"
    pead_context = build_pead_context(pd.read_csv(pead_path) if pead_path.is_file() else pd.DataFrame())

    review_src = out / "subject_taxonomy_review.csv"
    examples = pd.read_csv(review_src) if review_src.is_file() else None
    review = build_subject_review(counts, returns, volume, ranges, missing, examples)

    files = {
        "subject_impact_summary.csv": returns,
        "subject_volume_summary.csv": volume,
        "subject_range_summary.csv": ranges,
        "stock_subject_summary.csv": stock,
        "subject_missingness.csv": missing,
        "subject_evidence_ranking.csv": ranking,
        "subject_cluster_quality.csv": quality,
        "financial_results_pead_context.csv": pead_context,
        "subject_review.csv": review,
    }
    for name, frame in files.items():
        path = out / name
        frame.to_csv(path, index=False)
        print("Wrote", path, flush=True)

    print("Independent events", len(independent), flush=True)
    print(counts.to_string(index=False), flush=True)
    return {
        "counts": counts,
        "independent": independent,
        "returns": returns,
        "volume": volume,
        "ranges": ranges,
        "stock": stock,
        "missing": missing,
        "quality": quality,
        "ranking": ranking,
        "pead_context": pead_context,
        "review": review,
    }


def main(argv: list[str] | None = None) -> int:
    paths = parse_path_args(argv)
    run_subject_analysis(paths.data_dir, paths.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
