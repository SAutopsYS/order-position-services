"""Deterministic related-announcement clustering.

Same symbol plus the same normalized NEWSSUB, and a gap of 48 hours or less,
form one cluster. A second pass merges a financial-result announcement with a
nearby investor presentation or earnings transcript for the same symbol.

Different subjects on the same day are not merged unless that results-package
rule applies.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from src.taxonomy import is_financial_result
from src.text_normalize import normalize_text

MAX_GAP = timedelta(hours=48)
RESULTS_PACKAGE_SUBCATS = {
    "investor presentation",
    "earnings call transcript",
}


def _clean_label(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _is_results_package_companion(row: dict) -> bool:
    subcat = _clean_label(row.get("SUBCATNAME"))
    text = normalize_text(row.get("NEWSSUB")) + " " + normalize_text(row.get("HEADLINE"))
    return subcat in RESULTS_PACKAGE_SUBCATS or "investor presentation" in text or "earnings call transcript" in text


class _UnionFind:
    def __init__(self, labels: list[str]) -> None:
        self.parent = {label: label for label in labels}

    def find(self, label: str) -> str:
        while self.parent[label] != label:
            self.parent[label] = self.parent[self.parent[label]]
            label = self.parent[label]
        return label

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        # Deterministic parent: lexicographically smaller NEWSID.
        if root_left < root_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_left] = root_right


def assign_clusters(events: pd.DataFrame) -> pd.DataFrame:
    """Add cluster_id and cluster_size. event_id is NEWSID."""
    if events.empty:
        out = events.copy()
        out["cluster_id"] = []
        out["cluster_size"] = []
        return out

    frame = events.copy()
    frame["event_id"] = frame["NEWSID"].astype(str)
    frame["normalized_subject"] = frame["NEWSSUB"].map(normalize_text)
    frame["_ts"] = pd.to_datetime(frame["chosen_timestamp"], errors="coerce")
    frame = frame.sort_values(["symbol", "_ts", "event_id"], kind="mergesort")

    union = _UnionFind(frame["event_id"].tolist())

    for _, group in frame.groupby("symbol", sort=True):
        by_text = group.groupby("normalized_subject", sort=True)
        for _, text_group in by_text:
            ordered = text_group.sort_values(["_ts", "event_id"], kind="mergesort")
            previous = None
            for _, row in ordered.iterrows():
                if previous is not None and pd.notna(row["_ts"]) and pd.notna(previous["_ts"]):
                    if row["_ts"] - previous["_ts"] <= MAX_GAP:
                        union.union(previous["event_id"], row["event_id"])
                previous = row

        ordered = group.sort_values(["_ts", "event_id"], kind="mergesort")
        rows = ordered.to_dict(orient="records")
        for i, current in enumerate(rows):
            if not is_financial_result(current):
                continue
            for other in rows:
                if other["event_id"] == current["event_id"]:
                    continue
                if not _is_results_package_companion(other):
                    continue
                if pd.isna(current["_ts"]) or pd.isna(other["_ts"]):
                    continue
                if abs(current["_ts"] - other["_ts"]) <= MAX_GAP:
                    union.union(current["event_id"], other["event_id"])

    frame["cluster_id"] = frame["event_id"].map(lambda event_id: f"clu_{union.find(event_id)}")
    sizes = frame.groupby("cluster_id")["event_id"].transform("size")
    frame["cluster_size"] = sizes.astype(int)
    return frame.drop(columns=["_ts"])
