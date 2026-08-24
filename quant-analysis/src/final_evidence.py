"""Build Stage 6 evidence table, insights, and figure manifest from CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pead import SURPRISE_FIELD_TOKENS
from src.subject_analysis import STAGE2_SUBJECTS

REQUIRED_INSIGHT_FIELDS = (
    "insight_id",
    "topic",
    "observation",
    "direction",
    "magnitude",
    "n",
    "independent_event_count",
    "why_it_matters",
    "limitation",
    "source_output",
)

REQUIRED_EVIDENCE_FIELDS = (
    "evidence_id",
    "topic",
    "subject_group",
    "horizon",
    "observation",
    "direction",
    "magnitude",
    "n",
    "independent_event_count",
    "uncertainty",
    "limitation",
    "source_output",
)


def _fmt(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "unavailable"
    return f"{float(value):.{digits}f}"


def _row(frame: pd.DataFrame, **equals) -> pd.Series:
    part = frame
    for key, value in equals.items():
        part = part[part[key] == value]
    if part.empty:
        raise KeyError(str(equals))
    return part.iloc[0]


def build_candidate_insights(
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    ranges: pd.DataFrame,
    quality: pd.DataFrame,
    pead: pd.DataFrame,
    missing: pd.DataFrame,
    stock_subject: pd.DataFrame | None = None,
) -> pd.DataFrame:
    fr_vol = _row(volume, subject_group="financial_results", horizon="5m")
    cap_vol = _row(volume, subject_group="capital_structure", horizon="5m")
    fr_range = _row(ranges, subject_group="financial_results", horizon="session_close")
    cap_range = _row(ranges, subject_group="capital_structure", horizon="session_close")
    fr_sc = _row(returns, subject_group="financial_results", horizon="session_close")
    bus_sc = _row(returns, subject_group="business_updates", horizon="session_close")
    cap_q = _row(quality, subject_group="capital_structure")
    pead_pos_d1 = _row(pead, scope="pooled", initial_reaction="positive", horizon="d1")
    pead_neg_d1 = _row(pead, scope="pooled", initial_reaction="negative", horizon="d1")
    pead_pos_d3 = _row(pead, scope="pooled", initial_reaction="positive", horizon="d3")
    pead_neg_d3 = _row(pead, scope="pooled", initial_reaction="negative", horizon="d3")
    pead_all_d1 = _row(pead, scope="pooled", initial_reaction="all", horizon="d1")
    miss_60 = missing[(missing["horizon"] == "60m")]["missing_rate"].max()

    insights = [
        {
            "insight_id": "I1",
            "topic": "financial_results_volume",
            "observation": (
                f"Among {int(fr_vol['independent_event_count'])} independent financial_results events, "
                f"the median 5-minute volume ratio was {_fmt(fr_vol['median_volume_ratio'], 2)} "
                f"(mean {_fmt(fr_vol['mean_volume_ratio'], 2)}, n={int(fr_vol['n'])}), "
                f"compared with {_fmt(cap_vol['median_volume_ratio'], 2)} for capital_structure."
            ),
            "direction": "higher_volume_than_other_subjects",
            "magnitude": f"median_5m_volume_ratio={_fmt(fr_vol['median_volume_ratio'], 3)}",
            "n": int(fr_vol["n"]),
            "independent_event_count": int(fr_vol["independent_event_count"]),
            "why_it_matters": (
                "Financial-result windows show more trading activity than a ratio of 1, "
                "which is the prior-session clock-matched baseline."
            ),
            "limitation": (
                "The mean is larger than the median, so a few high-volume events pull the average. "
                "This is a volume comparison, not a return forecast."
            ),
            "source_output": "outputs/subject_volume_summary.csv",
        },
        {
            "insight_id": "I2",
            "topic": "financial_results_range",
            "observation": (
                f"Among {int(fr_range['independent_event_count'])} independent financial_results events, "
                f"the median session-close high-low range was {_fmt(fr_range['median_range'])} "
                f"(mean {_fmt(fr_range['mean_range'])}, n={int(fr_range['n'])}), "
                f"compared with {_fmt(cap_range['median_range'])} for capital_structure."
            ),
            "direction": "wider_intraday_range",
            "magnitude": f"median_session_close_range={_fmt(fr_range['median_range'])}",
            "n": int(fr_range["n"]),
            "independent_event_count": int(fr_range["independent_event_count"]),
            "why_it_matters": (
                "Result days show a wider same-session price range than routine capital-structure notices."
            ),
            "limitation": (
                "Range is window high / window low - 1. It is not a formal volatility model "
                "and does not isolate overnight gaps."
            ),
            "source_output": "outputs/subject_range_summary.csv",
        },
        {
            "insight_id": "I3",
            "topic": "financial_results_session_close",
            "observation": (
                f"Among {int(fr_sc['independent_event_count'])} independent financial_results events, "
                f"the mean session-close return was {_fmt(fr_sc['mean_return'])} "
                f"and the median was {_fmt(fr_sc['median_return'])} (n={int(fr_sc['n'])}). "
                f"The 95% interval was [{_fmt(fr_sc['ci95_lower'])}, {_fmt(fr_sc['ci95_upper'])}]."
            ),
            "direction": "slightly_negative_same_session",
            "magnitude": f"mean={_fmt(fr_sc['mean_return'])}; median={_fmt(fr_sc['median_return'])}",
            "n": int(fr_sc["n"]),
            "independent_event_count": int(fr_sc["independent_event_count"]),
            "why_it_matters": (
                "The same-session financial-result return is the conditioning input for PEAD, "
                "so its sign split matters more than the pooled mean."
            ),
            "limitation": (
                "The interval includes values near zero. Mean and median agree on sign but the "
                "effect is small and is not a trading rule."
            ),
            "source_output": "outputs/subject_impact_summary.csv",
        },
        {
            "insight_id": "I4",
            "topic": "price_conditioned_pead",
            "observation": (
                "This is a price-conditioned analysis based on observed post-announcement price reaction, "
                "not an earnings-surprise analysis. "
                f"Positive session-close reaction events (n={int(pead_pos_d1['n'])}) had mean D+1 "
                f"{_fmt(pead_pos_d1['mean_return'])} and mean D+3 {_fmt(pead_pos_d3['mean_return'])}. "
                f"Negative reaction events (n={int(pead_neg_d1['n'])}) had mean D+1 "
                f"{_fmt(pead_neg_d1['mean_return'])} and mean D+3 {_fmt(pead_neg_d3['mean_return'])}. "
                f"The unconditioned financial-result D+1 mean was {_fmt(pead_all_d1['mean_return'])} "
                f"(n={int(pead_all_d1['n'])}). Neutral initial-reaction events had 0 observations."
            ),
            "direction": "same_sign_continuation_in_sample",
            "magnitude": (
                f"positive_d1={_fmt(pead_pos_d1['mean_return'])}; "
                f"negative_d1={_fmt(pead_neg_d1['mean_return'])}"
            ),
            "n": int(pead_all_d1["n"]),
            "independent_event_count": int(pead_all_d1["n"]),
            "why_it_matters": (
                "The assignment requires price-conditioned PEAD on independent financial-result events. "
                "The sample shows later-session means that keep the sign of the event-day reaction."
            ),
            "limitation": (
                "Stock-wise sign cells are mostly n < 10 and are marked descriptive. "
                "No benchmark residual is removed. This pattern is descriptive, not causal, "
                "and is not an earnings-surprise result."
            ),
            "source_output": "outputs/pead_summary.csv",
        },
        {
            "insight_id": "I5",
            "topic": "capital_structure_cluster_concentration",
            "observation": (
                f"capital_structure has {int(cap_q['announcement_count'])} announcements but only "
                f"{int(cap_q['independent_event_count'])} independent events "
                f"(announcement-to-cluster ratio {_fmt(cap_q['announcement_to_cluster_ratio'], 2)}). "
                f"The largest cluster size is {int(cap_q['largest_cluster_size'])}."
            ),
            "direction": "repeated_notices_inflate_announcement_count",
            "magnitude": f"announcement_to_cluster_ratio={_fmt(cap_q['announcement_to_cluster_ratio'], 2)}",
            "n": int(cap_q["announcement_count"]),
            "independent_event_count": int(cap_q["independent_event_count"]),
            "why_it_matters": (
                "Treating every trading-window or allotment notice as independent would overstate "
                "the sample for this subject."
            ),
            "limitation": (
                "Cluster membership is rule-based on normalized subject text and a 48-hour gap. "
                "Independence is an implementation definition, not a statistical guarantee."
            ),
            "source_output": "outputs/subject_cluster_quality.csv",
        },
        {
            "insight_id": "I6",
            "topic": "business_updates_session_close",
            "observation": (
                f"Among {int(bus_sc['independent_event_count'])} independent business_updates events, "
                f"the mean session-close return was {_fmt(bus_sc['mean_return'])} "
                f"and the median was {_fmt(bus_sc['median_return'])} (n={int(bus_sc['n'])}). "
                f"The 95% interval was [{_fmt(bus_sc['ci95_lower'])}, {_fmt(bus_sc['ci95_upper'])}]."
            ),
            "direction": "small_negative_same_session",
            "magnitude": f"mean={_fmt(bus_sc['mean_return'])}",
            "n": int(bus_sc["n"]),
            "independent_event_count": int(bus_sc["independent_event_count"]),
            "why_it_matters": (
                "This is the largest subject group. The same-session mean is close to zero, "
                "which is useful context for the more active financial-result windows."
            ),
            "limitation": (
                "business_updates is a residual taxonomy bucket and mixes many notice types. "
                "A single mean should not be read as a common economic mechanism. "
                f"Intraday 60-minute missingness reaches {_fmt(miss_60, 3)} in some subjects."
            ),
            "source_output": "outputs/subject_impact_summary.csv",
        },
        {
            "insight_id": "I7",
            "topic": "stock_wise_pead_dispersion",
            "observation": (
                "Stock-wise financial-result PEAD means differ by symbol. "
                + (
                    "; ".join(
                        f"{row.symbol} D+1 mean={_fmt(row.mean_return)} (n={int(row.n)})"
                        for row in pead[
                            (pead["scope"] == "stock")
                            & (pead["initial_reaction"] == "all")
                            & (pead["horizon"] == "d1")
                        ].itertuples(index=False)
                    )
                    if not pead.empty
                    else "stock-wise D+1 rows unavailable"
                )
                + ". Sign-split stock cells are mostly n < 10."
            ),
            "direction": "not_uniform_across_stocks",
            "magnitude": "see stock_subject_summary and pead_summary stock rows",
            "n": int(pead_all_d1["n"]),
            "independent_event_count": int(pead_all_d1["n"]),
            "why_it_matters": (
                "A pooled PEAD average can hide stock-level differences in a five-name sample."
            ),
            "limitation": (
                "Per-stock financial-result samples are 12 to 24 independent events. "
                "Conditioned stock cells are descriptive_only and should not be ranked as strategies."
            ),
            "source_output": "outputs/pead_summary.csv",
        },
    ]
    return pd.DataFrame(insights)


def build_evidence_table(insights: pd.DataFrame, returns: pd.DataFrame, volume: pd.DataFrame, pead: pd.DataFrame) -> pd.DataFrame:
    fr_sc = _row(returns, subject_group="financial_results", horizon="session_close")
    fr_vol = _row(volume, subject_group="financial_results", horizon="5m")
    pead_pos = _row(pead, scope="pooled", initial_reaction="positive", horizon="d1")
    rows = []
    mapping = [
        ("E1", "I1", "financial_results", "5m", "volume"),
        ("E2", "I2", "financial_results", "session_close", "range"),
        ("E3", "I3", "financial_results", "session_close", "return"),
        ("E4", "I4", "financial_results", "d1", "price_conditioned_pead"),
        ("E5", "I5", "capital_structure", "session_close", "cluster_quality"),
        ("E6", "I6", "business_updates", "session_close", "return"),
        ("E7", "I7", "financial_results", "d1", "stock_wise_pead"),
    ]
    insight_map = insights.set_index("insight_id")
    extra_n = {
        "E1": int(fr_vol["n"]),
        "E3": int(fr_sc["n"]),
        "E4": int(pead_pos["n"]),
    }
    extra_uncert = {
        "E3": f"[{_fmt(fr_sc['ci95_lower'])}, {_fmt(fr_sc['ci95_upper'])}]",
        "E4": f"[{_fmt(pead_pos['ci95_lower'])}, {_fmt(pead_pos['ci95_upper'])}]",
    }
    for evidence_id, insight_id, subject, horizon, topic in mapping:
        insight = insight_map.loc[insight_id]
        rows.append(
            {
                "evidence_id": evidence_id,
                "topic": topic,
                "subject_group": subject,
                "horizon": horizon,
                "observation": insight["observation"],
                "direction": insight["direction"],
                "magnitude": insight["magnitude"],
                "n": extra_n.get(evidence_id, insight["n"]),
                "independent_event_count": insight["independent_event_count"],
                "uncertainty": extra_uncert.get(evidence_id, "see source summary"),
                "limitation": insight["limitation"],
                "source_output": insight["source_output"],
            }
        )
    return pd.DataFrame(rows)


def build_figure_manifest(figure_dir: Path) -> pd.DataFrame:
    rows = [
        {
            "figure_id": "F1",
            "filename": "fig01_subject_session_close_returns.png",
            "title": "Subject-group session-close returns",
            "purpose": "Compare same-session close-to-close returns across subjects",
            "source_output": "outputs/subject_impact_summary.csv",
            "main_variables": "mean_return, median_return, n",
            "notes": "Independent events. Not a ranking of best subjects.",
        },
        {
            "figure_id": "F2",
            "filename": "fig02_subject_d5_returns.png",
            "title": "Subject-group D+5 returns",
            "purpose": "Compare fifth subsequent session returns across subjects",
            "source_output": "outputs/subject_impact_summary.csv",
            "main_variables": "mean_return, median_return, n",
            "notes": "Corporate-actions mean and median diverge; treat that group cautiously.",
        },
        {
            "figure_id": "F3",
            "filename": "fig03_subject_volume_ratios.png",
            "title": "Subject-group median volume ratios",
            "purpose": "Compare trading activity versus the prior-session baseline",
            "source_output": "outputs/subject_volume_summary.csv",
            "main_variables": "median_volume_ratio",
            "notes": "A ratio of 1 matches the clock-time baseline. Medians are used because means are skewed.",
        },
        {
            "figure_id": "F4",
            "filename": "fig04_subject_session_close_range.png",
            "title": "Subject-group session-close range",
            "purpose": "Compare same-session high-low ranges",
            "source_output": "outputs/subject_range_summary.csv",
            "main_variables": "mean_range, median_range, n",
            "notes": "Range is window high / window low - 1.",
        },
        {
            "figure_id": "F5",
            "filename": "fig05_pead_price_conditioned.png",
            "title": "Price-Conditioned PEAD",
            "purpose": "Show later-session means by event-day price reaction",
            "source_output": "outputs/pead_summary.csv",
            "main_variables": "mean_return, ci95_lower, ci95_upper",
            "notes": "Neutral initial-reaction count is 0. Not earnings-surprise PEAD.",
        },
        {
            "figure_id": "F6",
            "filename": "fig06_stock_pead_descriptive.png",
            "title": "Stock-wise financial-result PEAD means",
            "purpose": "Show that pooled PEAD is not uniform across the five stocks",
            "source_output": "outputs/pead_summary.csv",
            "main_variables": "mean_return by symbol and horizon",
            "notes": "Descriptive sample results. Sign-split stock cells are small.",
        },
        {
            "figure_id": "F7",
            "filename": "fig07_subject_event_counts.png",
            "title": "Announcement count vs independent event count",
            "purpose": "Show why announcement rows are not the independent sample",
            "source_output": "outputs/subject_cluster_quality.csv",
            "main_variables": "announcement_count, independent_event_count",
            "notes": "capital_structure is the clearest gap between the two counts.",
        },
        {
            "figure_id": "F8",
            "filename": "fig08_financial_results_horizons.png",
            "title": "Financial-result returns across horizons",
            "purpose": "Show the event-impact path from 5 minutes through D+20",
            "source_output": "outputs/subject_impact_summary.csv",
            "main_variables": "mean_return, median_return, ci95",
            "notes": "Independent financial-result events only.",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["exists"] = [ (figure_dir / row["filename"]).is_file() for row in rows ]
    return frame


def has_surprise_fields(frame: pd.DataFrame) -> bool:
    lowered = [str(col).lower() for col in frame.columns]
    return any(any(token in col for token in SURPRISE_FIELD_TOKENS) for col in lowered)


def unsupported_subjects(frame: pd.DataFrame, column: str = "subject_group") -> list[str]:
    if column not in frame.columns:
        return []
    return sorted(set(frame[column].dropna().astype(str)) - set(STAGE2_SUBJECTS) - {"", "ALL"})
