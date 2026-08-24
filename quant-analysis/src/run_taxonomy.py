"""Build event clusters and the subject taxonomy from aligned announcements."""

from __future__ import annotations

import sys

import pandas as pd

from src.clustering import assign_clusters
from src.config import parse_path_args, resolve_paths
from src.data_loader import load_all_markets, load_announcements
from src.run_audit import build_alignment_table
from src.taxonomy import analysis_status, classify_subject, is_financial_result
from src.text_normalize import normalize_text


def _load_or_build_alignment(announcements: pd.DataFrame, paths) -> pd.DataFrame:
    alignment_path = paths.output_dir / "event_alignment_audit.csv"
    if alignment_path.is_file():
        return pd.read_csv(alignment_path)
    markets = load_all_markets(paths)
    return build_alignment_table(announcements, markets)


def build_event_table(announcements: pd.DataFrame, alignment: pd.DataFrame) -> pd.DataFrame:
    frame = announcements.copy()
    frame["event_id"] = frame["NEWSID"].astype(str)
    aligned = alignment.rename(columns={"announcement_id": "event_id"})
    aligned["event_id"] = aligned["event_id"].astype(str)
    keep = [
        "event_id",
        "chosen_timestamp",
        "timestamp_source",
        "effective_bar_timestamp",
        "effective_trading_date",
        "alignment_status",
        "exclusion_reason",
    ]
    merged = frame.merge(aligned[keep], on="event_id", how="left")
    merged["original_subject"] = merged["NEWSSUB"]
    merged["normalized_subject"] = merged["NEWSSUB"].map(normalize_text)
    merged["subject_group"] = merged.apply(lambda row: classify_subject(row.to_dict()), axis=1)
    merged["is_financial_result"] = merged.apply(lambda row: is_financial_result(row.to_dict()), axis=1)
    clustered = assign_clusters(merged)
    group_cluster_counts = (
        clustered.groupby("subject_group")["cluster_id"].nunique().rename("independent_event_count")
    )
    clustered = clustered.merge(group_cluster_counts, on="subject_group", how="left")
    clustered["analysis_status"] = clustered["independent_event_count"].map(analysis_status)
    return clustered


def subject_summary(events: pd.DataFrame) -> pd.DataFrame:
    summary = (
        events.groupby("subject_group", sort=True)
        .agg(
            announcement_count=("event_id", "size"),
            cluster_count=("cluster_id", "nunique"),
            independent_event_count=("cluster_id", "nunique"),
            financial_result_announcements=("is_financial_result", "sum"),
        )
        .reset_index()
    )
    summary["analysis_status"] = summary["independent_event_count"].map(analysis_status)
    return summary


def taxonomy_review(events: pd.DataFrame, examples_per_group: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, part in events.groupby("subject_group", sort=True):
        samples = part["original_subject"].dropna().astype(str).head(examples_per_group).tolist()
        row: dict[str, object] = {
            "subject_group": group,
            "announcement_count": len(part),
            "cluster_count": part["cluster_id"].nunique(),
        }
        for index in range(examples_per_group):
            row[f"example_{index + 1}"] = samples[index] if index < len(samples) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_quality(events: pd.DataFrame) -> dict[str, int]:
    sizes = events.groupby("cluster_id").size()
    return {
        "aligned_or_all_rows": len(events),
        "total_clusters": int(events["cluster_id"].nunique()),
        "singleton_clusters": int((sizes == 1).sum()),
        "multi_announcement_clusters": int((sizes > 1).sum()),
        "largest_cluster_size": int(sizes.max()) if len(sizes) else 0,
    }


def run_taxonomy(data_dir=None, output_dir=None) -> dict:
    if data_dir is None:
        paths = parse_path_args()
    else:
        paths = resolve_paths(data_dir, output_dir or "outputs")
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    announcements = load_announcements(paths.announcements_csv)
    alignment = _load_or_build_alignment(announcements, paths)
    events = build_event_table(announcements, alignment)
    summary = subject_summary(events)
    review = taxonomy_review(events)
    quality = cluster_quality(events)

    event_path = paths.output_dir / "event_clusters.csv"
    summary_path = paths.output_dir / "subject_taxonomy_summary.csv"
    review_path = paths.output_dir / "subject_taxonomy_review.csv"
    export_cols = [
        "event_id",
        "symbol",
        "chosen_timestamp",
        "timestamp_source",
        "effective_bar_timestamp",
        "effective_trading_date",
        "original_subject",
        "normalized_subject",
        "CATEGORYNAME",
        "SUBCATNAME",
        "subject_group",
        "is_financial_result",
        "cluster_id",
        "cluster_size",
        "analysis_status",
        "alignment_status",
        "exclusion_reason",
    ]
    events[export_cols].to_csv(event_path, index=False)
    summary.to_csv(summary_path, index=False)
    review.to_csv(review_path, index=False)

    print("Rows", len(events))
    print("Clusters", quality["total_clusters"])
    print("Singleton clusters", quality["singleton_clusters"])
    print("Multi-announcement clusters", quality["multi_announcement_clusters"])
    print("Largest cluster", quality["largest_cluster_size"])
    print("Financial-result announcements", int(events["is_financial_result"].sum()))
    print("Subject groups")
    print(summary.to_string(index=False))
    print("Wrote", event_path)
    print("Wrote", summary_path)
    print("Wrote", review_path)
    return {
        "events": events,
        "summary": summary,
        "review": review,
        "quality": quality,
    }


def main(argv: list[str] | None = None) -> int:
    paths = parse_path_args(argv)
    run_taxonomy(paths.data_dir, paths.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
