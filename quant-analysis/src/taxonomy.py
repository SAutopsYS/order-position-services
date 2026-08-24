"""Rule-based subject taxonomy and financial-result flags.

Priority (first match wins):
1. financial_results
2. credit_ratings
3. strategic_transactions
4. investor_communication
5. corporate_actions
6. board_governance
7. capital_structure
8. regulatory_compliance
9. business_updates
"""

from __future__ import annotations

from src.text_normalize import normalize_text

FINANCIAL_RESULT_PHRASES = (
    "financial result",
    "financial results",
    "quarterly result",
    "quarterly results",
    "annual result",
    "annual results",
    "audited result",
    "audited results",
    "unaudited result",
    "unaudited results",
    "standalone and consolidated financial",
    "consolidated and standalone unaudited",
)

INVESTOR_SUBCATS = {
    "analyst / investor meet",
    "earnings call transcript",
    "investor presentation",
    "newspaper publication",
}

STRATEGIC_SUBCATS = {
    "award of order / receipt of order",
    "acquisition",
    "memorandum of understanding /agreements",
    "diversification / disinvestment",
    "joint venture",
}

CORPORATE_ACTION_SUBCATS = {
    "dividend",
    "dividend updates",
    "record date",
    "book closure",
    "bonus",
    "sub-division / stock split",
}

GOVERNANCE_SUBCATS = {
    "agm",
    "board meeting",
    "outcome of board meeting",
    "outcome without intimation",
    "board meeting rescheduled",
    "postal ballot",
    "change in management",
    "change in directorate",
    "appointment of statutory auditor/s",
    "cessation",
    "retirement",
    "resignation of director",
    "resignation of company secretary / compliance officer",
    "appointment of company secretary / compliance officer",
    "resignation of chairman",
    "change in registered office address",
    "amendments to memorandum & articles of association",
}

CAPITAL_SUBCATS = {
    "allotment of esop / esps",
    "allotment of equity shares",
    "reg. 39 (3) - details of loss of certificate / duplicate certificate",
    "closure of trading window",
    "disclosures under reg. 29(2) of sebi (sast) regulations, 2011",
    "disclosures under reg. 29(1) of sebi (sast) regulations, 2011",
    "trading plan under sebi (pit) regulations, 2015",
}

REGULATORY_SUBCATS = {
    "certificate under reg. 74 (5) of sebi (dp) regulations, 2018",
    "reg. 34 (1) annual report",
    "business responsibility and sustainability reporting (brsr)",
    "reg.24(a)-annual secretarial compliance",
    "clarification",
    "reg. 40 (10) - pcs certificate for transfer / transmission / transposition",
    "reg. 54 - asset cover details",
    "monitoring agency report",
    "code of conduct under sebi (pit) regulations, 2015",
    "reg. 32 (1), (3) - statement of deviation & variation",
}

INFERENTIAL_MIN_INDEPENDENT_EVENTS = 10


def _clean_label(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _blob(row: dict) -> str:
    return " ".join(
        [
            normalize_text(row.get("NEWSSUB")),
            normalize_text(row.get("HEADLINE")),
            normalize_text(row.get("SUBCATNAME")),
        ]
    )


_NOT_RESULT_SUBCATS = {
    "analyst / investor meet",
    "earnings call transcript",
    "investor presentation",
    "newspaper publication",
    "press release / media release",
    "press release / media release (revised)",
    "closure of trading window",
}


def is_financial_result(row: dict) -> bool:
    """Conservative results flag.

    Official Result / Financial Results rows count. Board-meeting *outcomes*
    that publish results also count. Intimations, transcripts, presentations,
    press releases, and analyst meets do not.
    """
    category = _clean_label(row.get("CATEGORYNAME"))
    subcat = _clean_label(row.get("SUBCATNAME"))
    if category == "result" or subcat in {"financial results", "integrated filing (financial)"}:
        return True
    if subcat in _NOT_RESULT_SUBCATS:
        return False
    text = _blob(row)
    has_result_phrase = any(phrase in text for phrase in FINANCIAL_RESULT_PHRASES)
    if not has_result_phrase:
        return False
    if "board meeting intimation" in normalize_text(row.get("NEWSSUB")):
        return False
    return subcat in {"outcome of board meeting", "general"} or category == "company update"


def classify_subject(row: dict) -> str:
    category = _clean_label(row.get("CATEGORYNAME"))
    subcat = _clean_label(row.get("SUBCATNAME"))
    text = _blob(row)

    if is_financial_result(row):
        return "financial_results"
    if subcat == "credit rating" or "credit rating" in text:
        return "credit_ratings"
    if subcat in STRATEGIC_SUBCATS or category in {"corp. action"} and "acquisition" in text:
        return "strategic_transactions"
    if "award of order" in text or "receipt of order" in text:
        return "strategic_transactions"
    if "acquisition" in text:
        return "strategic_transactions"
    if subcat in INVESTOR_SUBCATS or "analyst" in text and "investor" in text:
        return "investor_communication"
    if "investor presentation" in text or "earnings call transcript" in text:
        return "investor_communication"
    if category in {"corp. action", "corp action"} or subcat in CORPORATE_ACTION_SUBCATS:
        return "corporate_actions"
    if category in {"board meeting", "agm/egm"} or subcat in GOVERNANCE_SUBCATS:
        return "board_governance"
    if "postal ballot" in text or "shareholder meeting" in text:
        return "board_governance"
    if category == "insider trading / sast" or subcat in CAPITAL_SUBCATS:
        return "capital_structure"
    if "loss of certificate" in text or "duplicate certificate" in text:
        return "capital_structure"
    if "esop" in text or "esps" in text:
        return "capital_structure"
    if subcat in REGULATORY_SUBCATS or category == "integrated filing":
        return "regulatory_compliance"
    if "annual report" in text or "secretarial compliance" in text:
        return "regulatory_compliance"
    return "business_updates"


def analysis_status(independent_event_count: int) -> str:
    if independent_event_count >= INFERENTIAL_MIN_INDEPENDENT_EVENTS:
        return "inferential"
    return "descriptive_only"
