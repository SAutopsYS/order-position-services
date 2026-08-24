import pandas as pd

from src.taxonomy import analysis_status, classify_subject, is_financial_result
from src.text_normalize import normalize_text


def test_normalization_is_stable() -> None:
    first = normalize_text("Announcement under Regulation 30 (LODR)-Credit Rating")
    second = normalize_text("Announcement under Regulation 30 (LODR)-Credit Rating")
    assert first == second
    assert first == "credit rating"


def test_original_subject_is_not_required_to_change() -> None:
    original = "Unaudited Financial Results For The Quarter Ended June 30, 2026."
    assert normalize_text(original) != original
    assert original.startswith("Unaudited")


def test_financial_result_from_category() -> None:
    row = {
        "CATEGORYNAME": "Result",
        "SUBCATNAME": "Financial Results",
        "NEWSSUB": "Financial Result For The Quarter Ended 30.06.2026",
        "HEADLINE": "Financial Result For The Quarter Ended 30.06.2026",
    }
    assert is_financial_result(row) is True
    assert classify_subject(row) == "financial_results"


def test_financial_result_from_text() -> None:
    row = {
        "CATEGORYNAME": "Company Update",
        "SUBCATNAME": "General",
        "NEWSSUB": "Unaudited Standalone And Consolidated Financial Results For The Quarter Ended June 30, 2026",
        "HEADLINE": "Unaudited financial results",
    }
    assert is_financial_result(row) is True


def test_earnings_transcript_is_not_a_financial_result() -> None:
    row = {
        "CATEGORYNAME": "Company Update",
        "SUBCATNAME": "Earnings Call Transcript",
        "NEWSSUB": "Announcement under Regulation 30 (LODR)-Earnings Call Transcript",
        "HEADLINE": "Transcript of Conference Call with Investors",
    }
    assert is_financial_result(row) is False
    assert classify_subject(row) == "investor_communication"


def test_certificate_loss_is_not_a_financial_result() -> None:
    row = {
        "CATEGORYNAME": "Company Update",
        "SUBCATNAME": "Reg. 39 (3) - Details of Loss of Certificate / Duplicate Certificate",
        "NEWSSUB": "Compliances-Reg. 39 (3) - Details of Loss of Certificate / Duplicate Certificate",
        "HEADLINE": "Loss of share certificate",
    }
    assert is_financial_result(row) is False
    assert classify_subject(row) == "capital_structure"


def test_small_groups_are_descriptive_only() -> None:
    assert analysis_status(9) == "descriptive_only"
    assert analysis_status(10) == "inferential"


def test_five_groups_from_actual_label_combinations() -> None:
    rows = [
        {"CATEGORYNAME": "Result", "SUBCATNAME": "Financial Results", "NEWSSUB": "Financial Results", "HEADLINE": "Financial Results"},
        {"CATEGORYNAME": "Company Update", "SUBCATNAME": "Analyst / Investor Meet", "NEWSSUB": "Analyst Meet", "HEADLINE": "Meet"},
        {"CATEGORYNAME": "Board Meeting", "SUBCATNAME": "Board Meeting", "NEWSSUB": "Board Meeting", "HEADLINE": "Board Meeting"},
        {"CATEGORYNAME": "Corp. Action", "SUBCATNAME": "Dividend", "NEWSSUB": "Dividend", "HEADLINE": "Dividend"},
        {"CATEGORYNAME": "Company Update", "SUBCATNAME": "Credit Rating", "NEWSSUB": "Credit Rating", "HEADLINE": "Rating"},
        {"CATEGORYNAME": "Company Update", "SUBCATNAME": "Award of Order / Receipt of Order", "NEWSSUB": "Award of Order", "HEADLINE": "Order"},
        {"CATEGORYNAME": "Company Update", "SUBCATNAME": "Allotment of ESOP / ESPS", "NEWSSUB": "ESOP", "HEADLINE": "ESOP"},
        {"CATEGORYNAME": "Company Update", "SUBCATNAME": "Certificate under Reg. 74 (5) of SEBI (DP) Regulations, 2018", "NEWSSUB": "Certificate", "HEADLINE": "Certificate"},
        {"CATEGORYNAME": "Others", "SUBCATNAME": "General", "NEWSSUB": "General Updates", "HEADLINE": "Update"},
    ]
    groups = {classify_subject(row) for row in rows}
    assert len(groups) >= 5
