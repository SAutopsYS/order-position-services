import pandas as pd

from src.clustering import assign_clusters
from src.taxonomy import classify_subject
from src.text_normalize import normalize_text


def _row(
    newsid: str,
    symbol: str,
    timestamp: str,
    newssub: str,
    category: str = "Company Update",
    subcat: str = "General",
    headline: str | None = None,
) -> dict:
    return {
        "NEWSID": newsid,
        "symbol": symbol,
        "chosen_timestamp": timestamp,
        "NEWSSUB": newssub,
        "HEADLINE": headline or newssub,
        "CATEGORYNAME": category,
        "SUBCATNAME": subcat,
    }


def test_similar_related_announcements_share_a_cluster() -> None:
    frame = pd.DataFrame(
        [
            _row("a1", "RELIANCE", "2024-01-02T10:00:00", "Credit Rating"),
            _row("a2", "RELIANCE", "2024-01-02T12:00:00", "Credit Rating"),
        ]
    )
    clustered = assign_clusters(frame)
    assert clustered["cluster_id"].nunique() == 1
    assert clustered["cluster_size"].tolist() == [2, 2]


def test_different_subjects_are_not_merged_just_because_they_are_close() -> None:
    frame = pd.DataFrame(
        [
            _row("a1", "RELIANCE", "2024-01-02T10:00:00", "Credit Rating", subcat="Credit Rating"),
            _row(
                "a2",
                "RELIANCE",
                "2024-01-02T10:05:00",
                "Compliances-Reg. 39 (3) - Details of Loss of Certificate / Duplicate Certificate",
                subcat="Reg. 39 (3) - Details of Loss of Certificate / Duplicate Certificate",
            ),
        ]
    )
    clustered = assign_clusters(frame)
    assert clustered["cluster_id"].nunique() == 2


def test_cluster_ids_are_deterministic() -> None:
    frame = pd.DataFrame(
        [
            _row("b2", "HAL", "2024-02-01T11:00:00", "Press Release / Media Release"),
            _row("b1", "HAL", "2024-02-01T10:00:00", "Press Release / Media Release"),
        ]
    )
    first = assign_clusters(frame)
    second = assign_clusters(frame.sample(frac=1.0, random_state=1).reset_index(drop=True))
    merged = first[["NEWSID", "cluster_id"]].merge(
        second[["NEWSID", "cluster_id"]], on="NEWSID", suffixes=("_a", "_b")
    )
    assert (merged["cluster_id_a"] == merged["cluster_id_b"]).all()


def test_every_row_has_a_cluster_id() -> None:
    frame = pd.DataFrame(
        [
            _row("a1", "NYKAA", "2024-03-01T09:00:00", "General Updates"),
            _row("a2", "RVNL", "2024-03-01T09:00:00", "General Updates"),
        ]
    )
    clustered = assign_clusters(frame)
    assert clustered["cluster_id"].notna().all()
    assert clustered["cluster_id"].str.startswith("clu_").all()


def test_cluster_size_is_correct() -> None:
    frame = pd.DataFrame(
        [
            _row("a1", "HDFCBANK", "2024-04-01T10:00:00", "Analyst Meet"),
            _row("a2", "HDFCBANK", "2024-04-01T11:00:00", "Analyst Meet"),
            _row("a3", "HDFCBANK", "2024-04-10T11:00:00", "Analyst Meet"),
        ]
    )
    clustered = assign_clusters(frame)
    sizes = clustered.set_index("NEWSID")["cluster_size"]
    assert int(sizes["a1"]) == 2
    assert int(sizes["a2"]) == 2
    assert int(sizes["a3"]) == 1


def test_output_preserves_original_event_ids() -> None:
    frame = pd.DataFrame([_row("keep-me", "HAL", "2024-05-01T10:00:00", "General Updates")])
    clustered = assign_clusters(frame)
    assert clustered["event_id"].tolist() == ["keep-me"]
    assert clustered["NEWSID"].tolist() == ["keep-me"]


def test_results_package_can_share_a_cluster() -> None:
    frame = pd.DataFrame(
        [
            _row(
                "r1",
                "RELIANCE",
                "2024-06-01T18:00:00",
                "Unaudited Financial Results For The Quarter Ended June 30, 2024",
                category="Result",
                subcat="Financial Results",
            ),
            _row(
                "r2",
                "RELIANCE",
                "2024-06-01T18:30:00",
                "Announcement under Regulation 30 (LODR)-Investor Presentation",
                subcat="Investor Presentation",
            ),
        ]
    )
    clustered = assign_clusters(frame)
    assert clustered["cluster_id"].nunique() == 1


def test_normalized_subject_is_stored_separately() -> None:
    original = "Announcement under Regulation 30 (LODR)-Credit Rating"
    assert normalize_text(original) != original
    row = {
        "CATEGORYNAME": "Company Update",
        "SUBCATNAME": "Credit Rating",
        "NEWSSUB": original,
        "HEADLINE": original,
    }
    assert classify_subject(row) == "credit_ratings"
