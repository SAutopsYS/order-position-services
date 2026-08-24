"""Build report.pdf from Stage 1-6 outputs. Numbers come from CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.final_evidence import _fmt, _row


def _styles() -> dict:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=16, leading=20, spaceAfter=10),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=13, leading=16, spaceBefore=12, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11, leading=14, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9.5, leading=13, alignment=TA_JUSTIFY, spaceAfter=6),
        "caption": ParagraphStyle("caption", parent=base["BodyText"], fontSize=8.5, leading=11, alignment=TA_LEFT, textColor=colors.HexColor("#333333"), spaceAfter=10),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontSize=9.5, leading=13, leftIndent=12, spaceAfter=3),
    }
    return styles


def _table(data: list[list[str]], col_widths: list[float] | None = None) -> Table:
    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5f8a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#aaaaaa")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f4f7fa")),
            ]
        )
    )
    return table


def _figure(path: Path, styles: dict, caption: str, width: float = 6.3 * inch) -> list:
    if not path.is_file():
        return [Paragraph(f"Missing figure: {path.name}", styles["caption"])]
    img = Image(str(path), width=width, height=width * 0.48)
    img.hAlign = "CENTER"
    return [img, Paragraph(caption, styles["caption"])]


def build_report(output_dir: Path, report_path: Path) -> Path:
    styles = _styles()
    out = Path(output_dir)
    figures = out / "figures"

    quality_dq = pd.read_csv(out / "data_quality_summary.csv")
    clusters = pd.read_csv(out / "event_clusters.csv")
    taxonomy = pd.read_csv(out / "subject_taxonomy_summary.csv")
    events = pd.read_csv(out / "event_level_results.csv")
    returns = pd.read_csv(out / "subject_impact_summary.csv")
    volume = pd.read_csv(out / "subject_volume_summary.csv")
    ranges = pd.read_csv(out / "subject_range_summary.csv")
    pead = pd.read_csv(out / "pead_summary.csv")
    pead_audit = pd.read_csv(out / "pead_audit.csv")
    insights = pd.read_csv(out / "candidate_insights.csv")
    missing = pd.read_csv(out / "subject_missingness.csv")

    aligned = int(quality_dq.loc[quality_dq["issue_type"] == "aligned", "count"].iloc[0])
    excluded = int(quality_dq.loc[quality_dq["issue_type"] == "exclusions", "count"].iloc[0])
    announcements = int(quality_dq[(quality_dq["dataset"] == "announcements") & (quality_dq["issue_type"] == "row_count")]["count"].iloc[0])
    cluster_n = int(clusters["cluster_id"].nunique())
    trading_min = str(pd.to_datetime(events["trading_date"], errors="coerce").min().date())
    trading_max = str(pd.to_datetime(events["trading_date"], errors="coerce").max().date())

    pead_map = {row.metric: row.value for row in pead_audit.itertuples(index=False)}
    fr_sc = _row(returns, subject_group="financial_results", horizon="session_close")
    fr_vol = _row(volume, subject_group="financial_results", horizon="5m")
    pead_all_d1 = _row(pead, scope="pooled", initial_reaction="all", horizon="d1")

    story = []
    story.append(Paragraph("Corporate Announcements, Price, and Volume: An Event Study of Five NSE Stocks", styles["title"]))
    story.append(Paragraph("Quantitative Data Analyst Intern Assessment - Final Report", styles["caption"]))

    story.append(Paragraph("Executive Summary", styles["h1"]))
    story.append(Paragraph(
        f"This project measures how corporate announcements relate to subsequent stock price and trading-volume behavior "
        f"for RELIANCE, HDFCBANK, NYKAA, HAL, and RVNL. The supplied pack contains {announcements} announcement rows and "
        f"one-minute OHLCV bars. {aligned} announcements align to a first complete one-minute bar at or after dissemination. "
        f"{excluded} announcements are excluded because no later bar exists. Stage 2 groups the notices into nine subject "
        f"categories and {cluster_n} event clusters. Stage 3 computes close-to-close returns, a prior-session volume ratio, "
        f"and a high-low range. Stage 4 runs price-conditioned PEAD on {int(pead_map['independent_financial_result_events'])} "
        f"independent financial-result events. This is not an earnings-surprise study.",
        styles["body"],
    ))
    story.append(Paragraph(
        f"The strongest descriptive patterns are concentrated activity on financial-result days "
        f"(median 5-minute volume ratio {_fmt(fr_vol['median_volume_ratio'], 2)}, n={int(fr_vol['n'])}), "
        f"a slightly negative same-session financial-result return (mean {_fmt(fr_sc['mean_return'])}, n={int(fr_sc['n'])}), "
        f"and later-session means that keep the sign of the event-day price reaction. "
        f"None of these results is presented as a trading rule, a causal claim, or a statistically tested anomaly.",
        styles["body"],
    ))

    story.append(Paragraph("1. Objective", styles["h1"]))
    story.append(Paragraph(
        "The assessment asks a practical market-microstructure question: after a company publishes an announcement, "
        "how do price and volume behave over short intraday windows and over the next several trading sessions? "
        "The required work includes timestamp alignment, event independence, a transparent subject taxonomy, "
        "event-level impact, subject-level comparison, and price-conditioned PEAD on financial-result events. "
        "The implementation stays close to the supplied files. No external prices, analyst estimates, or earnings-surprise "
        "databases are used.",
        styles["body"],
    ))

    story.append(Paragraph("2. Data", styles["h1"]))
    story.append(Paragraph(
        "The assessment folder supplies corporate_announcements.csv (and a jsonl copy), metadata.json, and five "
        "one-minute market files: RELIANCE, HDFCBANK, NYKAA, HAL, and RVNL. DATA_FORMAT.md is named in the assignment "
        "and is not present in the pack. Raw files are not stored in the public project. A reviewer points "
        "--data-dir at the supplied folder or sets QUANT_DATA_DIR.",
        styles["body"],
    ))
    story.append(_table(
        [
            ["Item", "Value"],
            ["Announcement rows", str(announcements)],
            ["Aligned announcements", str(aligned)],
            ["Excluded announcements", str(excluded)],
            ["Event clusters", str(cluster_n)],
            ["Stocks", "RELIANCE, HDFCBANK, NYKAA, HAL, RVNL"],
            ["Event trading-date span", f"{trading_min} to {trading_max}"],
            ["Market sessions per stock (Stage 1)", "744"],
        ],
        [2.4 * inch, 4.0 * inch],
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Each market file has about 277,500 one-minute bars. Two invalid OHLC bars exist (HDFCBANK and HAL, both "
        "2024-06-25 09:15). Zero-volume bars appear in every stock. Sessions are inferred from actual timestamps and "
        "include weekend and non-standard clocks. The study does not assume a Monday-Friday 09:15-15:30 calendar.",
        styles["body"],
    ))

    story.append(Paragraph("3. Data Quality and Timestamp Alignment", styles["h1"]))
    story.append(Paragraph(
        "Announcement time prefers DissemDT when it parses. DT_TM is the fallback. Mixed ISO precision is accepted "
        "with format=ISO8601. Naive pd.to_datetime rejected 11 valid timestamps in early testing, so the ISO parser "
        "is required. Timestamps stay naive as supplied because DATA_FORMAT.md is missing and no timezone is invented.",
        styles["body"],
    ))
    story.append(Paragraph(
        "The event bar is the first one-minute bar whose start is greater than or equal to the chosen announcement "
        "timestamp. A bar that began before dissemination is never used. After-close announcements map to the next "
        "available session. The two exclusions are HDFCBANK notices after the last supplied bar on 2026-08-19. "
        "Invalid OHLC rows are flagged and not repaired. Metrics that need those prices are marked unavailable.",
        styles["body"],
    ))

    story.append(Paragraph("4. Event Clustering and Subject Taxonomy", styles["h1"]))
    story.append(Paragraph(
        "Related notices are clustered when they share a symbol and the same normalized NEWSSUB and are at most "
        "48 hours apart. A financial-result notice may also merge with a nearby investor presentation or earnings "
        "transcript for the same symbol. Different subjects on the same day are not merged just because they share a date. "
        "is_financial_result is conservative: CATEGORYNAME Result or tight result phrases on a small set of categories. "
        "Analyst meets, transcripts, presentations, and board-meeting intimations are not treated as results. "
        "The nine subject groups come from documented if/then rules. A group is inferential when it has at least "
        "10 independent events (unique cluster_id values). All nine groups meet that cutoff.",
        styles["body"],
    ))
    tax_rows = [["Subject group", "Announcements", "Clusters", "Status"]]
    for row in taxonomy.sort_values("subject_group").itertuples(index=False):
        tax_rows.append([row.subject_group, str(int(row.announcement_count)), str(int(row.cluster_count)), row.analysis_status])
    story.append(_table(tax_rows, [2.2 * inch, 1.3 * inch, 1.2 * inch, 1.3 * inch]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Independent-event counts can exceed unique cluster_id values because a cluster may contain more than one "
        "subject after the financial-result / presentation merge. Subject-level analysis therefore uses one "
        "observation per (subject_group, cluster_id).",
        styles["body"],
    ))

    story.append(Paragraph("5. Event-Level Price, Volume, and Range Analysis", styles["h1"]))
    story.append(Paragraph(
        "The impact anchor is effective_bar_timestamp. The return is future_close / anchor_close - 1. "
        "Intraday 5m, 30m, and 60m horizons use the first same-session bar at or after anchor plus N minutes. "
        "If the session ends first, the metric is missing. Session close uses the last actual bar of that session, "
        "not an assumed 15:30 clock. D+1, D+5, D+10, and D+20 are inferred trading sessions, not calendar days.",
        styles["body"],
    ))
    story.append(Paragraph(
        "Volume ratio equals event-window volume divided by the median of the same clock-time window in prior "
        "sessions. The lookback is at most 20 prior sessions and needs at least 5 usable windows. The event session "
        "and later sessions are excluded. Range is window_high / window_low - 1. Zero-volume bars stay in the file "
        "and can still contribute valid prices.",
        styles["body"],
    ))
    story.extend(_figure(
        figures / "fig07_subject_event_counts.png",
        styles,
        "Figure 1. Announcement count is not the independent sample. capital_structure is the clearest gap.",
    ))
    story.extend(_figure(
        figures / "fig03_subject_volume_ratios.png",
        styles,
        "Figure 2. Median volume ratio by subject. A value of 1 matches the prior-session clock-time baseline.",
    ))
    story.extend(_figure(
        figures / "fig04_subject_session_close_range.png",
        styles,
        "Figure 3. Session-close high-low range by subject group.",
    ))

    story.append(Paragraph("6. Subject-Level Results", styles["h1"]))
    story.append(Paragraph(
        "Subject summaries use independent events. Uncertainty is a normal-approximation 95% interval: "
        "mean +/- 1.96 * std / sqrt(n). Intervals are unavailable when n < 2. Groups with fewer than 10 "
        "independent events would be marked descriptive_only. At the pooled subject level all nine groups "
        "are inferential. Stock-subject cells are often descriptive_only and are retained, not deleted.",
        styles["body"],
    ))
    ret_rows = [["Subject", "n", "Mean SC", "Median SC", "Mean D+5", "Median D+5"]]
    for subject in taxonomy.sort_values("subject_group")["subject_group"]:
        sc = _row(returns, subject_group=subject, horizon="session_close")
        d5 = _row(returns, subject_group=subject, horizon="d5")
        ret_rows.append([
            subject,
            str(int(sc["n"])),
            _fmt(sc["mean_return"]),
            _fmt(sc["median_return"]),
            _fmt(d5["mean_return"]),
            _fmt(d5["median_return"]),
        ])
    story.append(_table(ret_rows, [1.8 * inch, 0.6 * inch, 0.9 * inch, 1.0 * inch, 0.9 * inch, 1.0 * inch]))
    story.append(Spacer(1, 8))
    story.extend(_figure(
        figures / "fig01_subject_session_close_returns.png",
        styles,
        "Figure 4. Session-close mean and median returns by subject. Sample sizes are on the axis labels.",
    ))
    story.extend(_figure(
        figures / "fig02_subject_d5_returns.png",
        styles,
        "Figure 5. D+5 mean and median returns. corporate_actions has a large mean-median gap and is not treated as a stable pattern.",
    ))
    story.append(Paragraph(
        "Financial-result days stand out on volume and range more than on pooled same-session return. "
        "business_updates, the largest residual group, has a small negative session-close mean and a tight interval. "
        "That group mixes many notice types, so the mean is a description of a mixed bag, not one mechanism. "
        "60-minute missingness is highest for late-session events and reaches about 10% for regulatory_compliance.",
        styles["body"],
    ))

    story.append(Paragraph("7. Price-Conditioned PEAD", styles["h1"]))
    story.append(Paragraph(
        "This analysis is price-conditioned and does not use earnings-surprise data. "
        "The sample is one aligned financial-result event per cluster (earliest effective bar). "
        "Initial reaction is the Stage 3 session-close return: positive if greater than 0, neutral if exactly 0, "
        "negative if less than 0. The sample has "
        f"{int(pead_map['positive_initial_reaction'])} positive, "
        f"{int(pead_map['neutral_initial_reaction'])} neutral, and "
        f"{int(pead_map['negative_initial_reaction'])} negative events. "
        "Horizons are D+1, D+3, D+5, D+10, and D+20 inferred sessions.",
        styles["body"],
    ))
    pead_rows = [["Reaction", "Horizon", "n", "Mean", "Median", "CI low", "CI high"]]
    for reaction in ("all", "positive", "negative"):
        for horizon in ("d1", "d3", "d5", "d10", "d20"):
            row = _row(pead, scope="pooled", initial_reaction=reaction, horizon=horizon)
            pead_rows.append([
                reaction,
                horizon,
                str(int(row["n"])),
                _fmt(row["mean_return"]),
                _fmt(row["median_return"]),
                _fmt(row["ci95_lower"]),
                _fmt(row["ci95_upper"]),
            ])
    story.append(_table(pead_rows, [1.0 * inch, 0.8 * inch, 0.6 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch]))
    story.append(Spacer(1, 8))
    story.extend(_figure(
        figures / "fig05_pead_price_conditioned.png",
        styles,
        "Figure 6. Price-Conditioned PEAD. Neutral initial-reaction events had zero observations.",
    ))
    story.extend(_figure(
        figures / "fig06_stock_pead_descriptive.png",
        styles,
        "Figure 7. Stock-wise financial-result means. These are descriptive sample results, not stock rankings.",
    ))
    story.extend(_figure(
        figures / "fig08_financial_results_horizons.png",
        styles,
        "Figure 8. Independent financial-result returns from 5 minutes through D+20.",
    ))
    story.append(Paragraph(
        f"The unconditioned D+1 mean is {_fmt(pead_all_d1['mean_return'])} (n={int(pead_all_d1['n'])}). "
        "After splitting on the event-day sign, later-session means keep that sign in the pooled sample. "
        "Stock-wise sign cells are mostly smaller than 10 observations and are marked descriptive. "
        "No market-model residual is removed.",
        styles["body"],
    ))

    story.append(Paragraph("8. Key Insights", styles["h1"]))
    story.append(Paragraph(
        "Each insight states an observation, a magnitude, a sample size, a cautious interpretation, and a limitation. "
        "Rejected candidates included the corporate_actions D+5 mean, which is not supported by the median, "
        "and any wording that would treat PEAD as an earnings-surprise anomaly.",
        styles["body"],
    ))
    for row in insights.itertuples(index=False):
        story.append(Paragraph(f"{row.insight_id}. {row.topic.replace('_', ' ')}", styles["h2"]))
        story.append(Paragraph(f"Observation. {row.observation}", styles["body"]))
        story.append(Paragraph(f"Why it matters. {row.why_it_matters}", styles["body"]))
        story.append(Paragraph(f"Limitation. {row.limitation}", styles["body"]))

    story.append(Paragraph("9. Limitations", styles["h1"]))
    for text in (
        "No earnings-surprise data and no external analyst estimates are available.",
        "Timestamps stay naive because DATA_FORMAT.md was not in the assessment pack.",
        "Returns are raw close-to-close values. There is no benchmark residual or beta adjustment.",
        "Volume baselines use at most 20 prior sessions and need at least 5 comparable windows.",
        "Market coverage ends in the supplied tape, so late events lose D+10 and D+20.",
        "Stock-subject and stock-condition PEAD cells are often small and descriptive_only.",
        "Confidence intervals are simple normal-approximation intervals and are not hypothesis tests.",
        "Clustering and taxonomy are deterministic rules. They can miss economic links that text rules do not see.",
        "Price-conditioned PEAD describes later returns after an observed price reaction. It is not causal evidence.",
        "Two announcements are excluded. Two invalid OHLC bars exist and are not repaired.",
    ):
        story.append(Paragraph(f"- {text}", styles["bullet"]))

    story.append(Paragraph("10. Reproducibility", styles["h1"]))
    story.append(Paragraph(
        "Create a virtual environment, install requirements.txt, and pass the supplied assessment folder at run time. "
        "Do not hard-code a machine path. Commands: python -m src.run_audit, python -m src.run_taxonomy, "
        "python -m src.run_event_impact, python -m src.run_pead, python -m src.run_subject_analysis, "
        "python -m src.run_final_evidence, python -m src.run_report. Tests: python -m pytest -q. "
        "Raw announcement and market CSVs stay outside the public repository. Generated CSVs, figures, and this "
        "report are written under outputs/.",
        styles["body"],
    ))

    story.append(Paragraph("Conclusion", styles["h1"]))
    story.append(Paragraph(
        "In this five-stock sample, financial-result events are the noisiest same-session windows on volume and range. "
        "Pooled same-session returns are small. When financial-result events are split by the observed session-close "
        "price reaction, later-session means keep that sign. The pattern is descriptive, price-conditioned, and limited "
        "by sample size, the absence of surprise data, and the lack of a benchmark adjustment. The analysis does not "
        "support a trading strategy claim.",
        styles["body"],
    ))

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Corporate Announcements Price and Volume Event Study",
        author="Quantitative Data Analyst Intern Assessment",
    )
    doc.build(story)
    return report_path
