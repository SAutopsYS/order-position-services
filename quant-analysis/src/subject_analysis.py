"""Subject-level aggregation of Stage 3 event metrics.

Stage 2 subject_group names are the source of truth. Returns, volume
ratios, and ranges are not recomputed. One independent observation is
used per (subject_group, cluster_id). Announcement counts stay separate
from cluster counts.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.event_impact import CI_Z, mean_confidence_interval
from src.taxonomy import analysis_status

STAGE2_SUBJECTS = (
    "business_updates",
    "capital_structure",
    "investor_communication",
    "board_governance",
    "strategic_transactions",
    "regulatory_compliance",
    "financial_results",
    "credit_ratings",
    "corporate_actions",
)

RETURN_HORIZONS = (
    "5m",
    "30m",
    "60m",
    "session_close",
    "d1",
    "d5",
    "d10",
    "d20",
)
VOLUME_HORIZONS = ("5m", "30m", "60m", "session_close")
RANGE_HORIZONS = VOLUME_HORIZONS


def select_independent_subject_events(events: pd.DataFrame) -> pd.DataFrame:
    """One row per subject_group and cluster_id. Earliest bar wins."""
    if events.empty:
        return events.copy()
    frame = events.copy()
    frame["_sort_bar"] = pd.to_datetime(frame["effective_bar_timestamp"], errors="coerce")
    frame["_sort_id"] = frame["event_id"].astype(str)
    frame = frame.sort_values(
        ["subject_group", "cluster_id", "_sort_bar", "_sort_id"],
        kind="mergesort",
    )
    chosen = frame.groupby(["subject_group", "cluster_id"], as_index=False, sort=False).first()
    return chosen.drop(columns=["_sort_bar", "_sort_id"], errors="ignore")


def subject_counts(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, part in events.groupby("subject_group", dropna=False):
        cluster_count = int(part["cluster_id"].nunique())
        rows.append(
            {
                "subject_group": subject,
                "announcement_count": int(len(part)),
                "cluster_count": cluster_count,
                "independent_event_count": cluster_count,
                "analysis_status": analysis_status(cluster_count),
            }
        )
    return pd.DataFrame(rows).sort_values("subject_group").reset_index(drop=True)


def summarize_metric(values: pd.Series, value_name: str) -> dict[str, object]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n = int(len(clean))
    row = {
        "n": n,
        f"mean_{value_name}": float(clean.mean()) if n else np.nan,
        f"median_{value_name}": float(clean.median()) if n else np.nan,
        f"std_{value_name}": float(clean.std(ddof=1)) if n > 1 else np.nan,
        "standard_error": np.nan,
        "ci95_lower": np.nan,
        "ci95_upper": np.nan,
    }
    if n >= 2:
        mean, low, high, error = mean_confidence_interval(clean)
        row["standard_error"] = float(clean.std(ddof=1) / np.sqrt(n))
        row["ci95_lower"] = low
        row["ci95_upper"] = high
        if error is None and mean is not None:
            row[f"mean_{value_name}"] = mean
    return row


def _usable(part: pd.DataFrame, horizon: str) -> pd.DataFrame:
    flag = f"usable_{horizon}"
    if flag not in part.columns:
        return part.iloc[0:0]
    return part[part[flag] == True]  # noqa: E712


def build_subject_metric_summary(
    independent: pd.DataFrame,
    counts: pd.DataFrame,
    horizons: tuple[str, ...],
    column_prefix: str,
    value_name: str,
) -> pd.DataFrame:
    count_map = counts.set_index("subject_group")
    rows = []
    for subject, part in independent.groupby("subject_group", dropna=False):
        status = count_map.loc[subject, "analysis_status"] if subject in count_map.index else analysis_status(int(part["cluster_id"].nunique()))
        independent_n = int(part["cluster_id"].nunique())
        for horizon in horizons:
            usable = _usable(part, horizon)
            metric = summarize_metric(usable[f"{column_prefix}_{horizon}"], value_name)
            rows.append(
                {
                    "subject_group": subject,
                    "horizon": horizon,
                    "independent_event_count": independent_n,
                    "analysis_status": status,
                    **metric,
                }
            )
    return pd.DataFrame(rows)


def build_stock_subject_summary(independent: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (symbol, subject), part in independent.groupby(["symbol", "subject_group"], dropna=False):
        independent_n = int(part["cluster_id"].nunique())
        status = analysis_status(independent_n)
        for horizon in RETURN_HORIZONS:
            usable = _usable(part, horizon)
            metric = summarize_metric(usable[f"return_{horizon}"], "return")
            rows.append(
                {
                    "symbol": symbol,
                    "subject_group": subject,
                    "horizon": horizon,
                    "independent_event_count": independent_n,
                    "analysis_status": status,
                    "n": metric["n"],
                    "mean_return": metric["mean_return"],
                    "median_return": metric["median_return"],
                    "standard_error": metric["standard_error"],
                    "ci95_lower": metric["ci95_lower"],
                    "ci95_upper": metric["ci95_upper"],
                }
            )
    return pd.DataFrame(rows)


def build_missingness(independent: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, part in independent.groupby("subject_group", dropna=False):
        total = int(len(part))
        for horizon in RETURN_HORIZONS:
            usable = int(_usable(part, horizon).shape[0])
            missing = total - usable
            rows.append(
                {
                    "subject_group": subject,
                    "horizon": horizon,
                    "total_events": total,
                    "usable_events": usable,
                    "missing_events": missing,
                    "missing_rate": float(missing / total) if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_cluster_quality(events: pd.DataFrame, independent: pd.DataFrame) -> pd.DataFrame:
    counts = subject_counts(events)
    sizes = events.groupby(["subject_group", "cluster_id"], dropna=False)["cluster_size"].max()
    size_stats = sizes.groupby("subject_group").agg(largest_cluster_size="max", median_cluster_size="median")
    rows = []
    for _, count in counts.iterrows():
        subject = count["subject_group"]
        part = events[events["subject_group"] == subject]
        subject_sizes = sizes.xs(subject) if subject in sizes.index.get_level_values(0) else pd.Series(dtype=float)
        largest = int(size_stats.loc[subject, "largest_cluster_size"]) if subject in size_stats.index else 0
        in_largest = int((part["cluster_size"] == largest).sum()) if largest else 0
        ann_mean = _mean_usable(part, "return_session_close", "session_close")
        ind_part = independent[independent["subject_group"] == subject]
        ind_mean = _mean_usable(ind_part, "return_session_close", "session_close")
        rows.append(
            {
                "subject_group": subject,
                "announcement_count": count["announcement_count"],
                "cluster_count": count["cluster_count"],
                "independent_event_count": count["independent_event_count"],
                "analysis_status": count["analysis_status"],
                "largest_cluster_size": largest,
                "median_cluster_size": float(size_stats.loc[subject, "median_cluster_size"]) if subject in size_stats.index else np.nan,
                "announcements_in_largest_cluster": in_largest,
                "share_in_largest_cluster": float(in_largest / count["announcement_count"]) if count["announcement_count"] else np.nan,
                "announcement_to_cluster_ratio": float(count["announcement_count"] / count["cluster_count"]) if count["cluster_count"] else np.nan,
                "mean_session_close_announcement_level": ann_mean,
                "mean_session_close_independent_level": ind_mean,
            }
        )
    return pd.DataFrame(rows)


def _mean_usable(part: pd.DataFrame, column: str, horizon: str) -> float:
    if part.empty or column not in part.columns:
        return np.nan
    usable = _usable(part, horizon)
    values = pd.to_numeric(usable[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def build_evidence_ranking(return_summary: pd.DataFrame, volume_summary: pd.DataFrame, range_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in RETURN_HORIZONS:
        part = return_summary[return_summary["horizon"] == horizon].copy()
        part["abs_mean"] = part["mean_return"].abs()
        part = part.sort_values("abs_mean", ascending=False, kind="mergesort")
        for rank, (_, row) in enumerate(part.iterrows(), start=1):
            rows.append(
                {
                    "rank_family": f"largest_abs_mean_return_{horizon}",
                    "rank": rank,
                    "subject_group": row["subject_group"],
                    "horizon": horizon,
                    "metric": "abs_mean_return",
                    "value": row["abs_mean"],
                    "signed_mean": row["mean_return"],
                    "n": row["n"],
                    "analysis_status": row["analysis_status"],
                }
            )
    for horizon in VOLUME_HORIZONS:
        part = volume_summary[volume_summary["horizon"] == horizon].copy()
        part = part.sort_values("mean_volume_ratio", ascending=False, kind="mergesort")
        for rank, (_, row) in enumerate(part.iterrows(), start=1):
            rows.append(
                {
                    "rank_family": f"highest_mean_volume_ratio_{horizon}",
                    "rank": rank,
                    "subject_group": row["subject_group"],
                    "horizon": horizon,
                    "metric": "mean_volume_ratio",
                    "value": row["mean_volume_ratio"],
                    "signed_mean": row["mean_volume_ratio"],
                    "n": row["n"],
                    "analysis_status": row["analysis_status"],
                }
            )
    for horizon in RANGE_HORIZONS:
        part = range_summary[range_summary["horizon"] == horizon].copy()
        part = part.sort_values("mean_range", ascending=False, kind="mergesort")
        for rank, (_, row) in enumerate(part.iterrows(), start=1):
            rows.append(
                {
                    "rank_family": f"highest_mean_range_{horizon}",
                    "rank": rank,
                    "subject_group": row["subject_group"],
                    "horizon": horizon,
                    "metric": "mean_range",
                    "value": row["mean_range"],
                    "signed_mean": row["mean_range"],
                    "n": row["n"],
                    "analysis_status": row["analysis_status"],
                }
            )
    return pd.DataFrame(rows)


def build_pead_context(pead_summary: pd.DataFrame) -> pd.DataFrame:
    if pead_summary.empty:
        return pead_summary.copy()
    frame = pead_summary.copy()
    if "scope" in frame.columns:
        frame = frame[frame["scope"] == "pooled"]
    frame = frame[frame["initial_reaction"].isin(["all", "positive", "neutral", "negative"])]
    out = frame[
        [
            col
            for col in (
                "initial_reaction",
                "horizon",
                "n",
                "mean_return",
                "median_return",
                "ci95_lower",
                "ci95_upper",
                "sample_note",
            )
            if col in frame.columns
        ]
    ].copy()
    out.insert(0, "subject_group", "financial_results")
    out.insert(1, "conditioning", "price_conditioned_session_close")
    return out.reset_index(drop=True)


def build_subject_review(
    counts: pd.DataFrame,
    return_summary: pd.DataFrame,
    volume_summary: pd.DataFrame,
    range_summary: pd.DataFrame,
    missingness: pd.DataFrame,
    examples: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = []
    for _, count in counts.iterrows():
        subject = count["subject_group"]
        ret = return_summary[return_summary["subject_group"] == subject]
        vol = volume_summary[volume_summary["subject_group"] == subject]
        rng = range_summary[range_summary["subject_group"] == subject]
        miss = missingness[missingness["subject_group"] == subject]
        example_text = ""
        if examples is not None and not examples.empty:
            match = examples[examples["subject_group"] == subject]
            if not match.empty:
                cols = [c for c in match.columns if c.startswith("example_")]
                example_text = " | ".join(str(match.iloc[0][c]) for c in cols if pd.notna(match.iloc[0][c]))
        miss_bits = [
            f"{row.horizon}:{row.missing_events}/{row.total_events}"
            for row in miss.itertuples(index=False)
        ]
        rows.append(
            {
                "subject_group": subject,
                "independent_event_count": count["independent_event_count"],
                "announcement_count": count["announcement_count"],
                "cluster_count": count["cluster_count"],
                "analysis_status": count["analysis_status"],
                "representative_example_subjects": example_text,
                "mean_session_close_return": _lookup_mean(ret, "session_close", "mean_return"),
                "mean_d1_return": _lookup_mean(ret, "d1", "mean_return"),
                "mean_d5_return": _lookup_mean(ret, "d5", "mean_return"),
                "mean_d20_return": _lookup_mean(ret, "d20", "mean_return"),
                "mean_volume_ratio_5m": _lookup_mean(vol, "5m", "mean_volume_ratio"),
                "mean_volume_ratio_session_close": _lookup_mean(vol, "session_close", "mean_volume_ratio"),
                "mean_range_session_close": _lookup_mean(rng, "session_close", "mean_range"),
                "missingness_summary": "; ".join(miss_bits),
            }
        )
    return pd.DataFrame(rows)


def _lookup_mean(frame: pd.DataFrame, horizon: str, column: str) -> float:
    match = frame[frame["horizon"] == horizon]
    if match.empty:
        return np.nan
    return match.iloc[0][column]


def unexpected_subjects(subjects: pd.Series) -> list[str]:
    found = {str(value) for value in subjects.dropna().unique()}
    return sorted(found - set(STAGE2_SUBJECTS))
