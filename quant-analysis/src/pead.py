"""Price-conditioned PEAD for independent financial-result events.

This is not earnings-surprise PEAD. No analyst estimates are used.
The conditioning variable is the Stage 3 session-close return.

Independent observations use one aligned financial-result event per
cluster. The representative is the earliest aligned financial-result
row in that cluster.

PEAD horizons are inferred trading sessions, not calendar days:

    future_session_close / anchor_close - 1
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.event_impact import CI_Z, MarketIndex, future_session_return, mean_confidence_interval

PEAD_HORIZONS = {"d1": 1, "d3": 3, "d5": 5, "d10": 10, "d20": 20}
DESCRIPTIVE_N = 10
SURPRISE_FIELD_TOKENS = ("surprise", "eps", "estimate", "analyst", "consensus")


def is_true_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def classify_initial_reaction(session_close_return: object) -> str:
    value = pd.to_numeric(session_close_return, errors="coerce")
    if pd.isna(value):
        return "unavailable"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def select_independent_financial_events(clusters: pd.DataFrame) -> pd.DataFrame:
    """One aligned financial-result event per cluster. Earliest bar wins."""
    if clusters.empty:
        return clusters.copy()
    frame = clusters.copy()
    financial = frame[frame["is_financial_result"].map(is_true_flag)]
    aligned = financial[financial["alignment_status"].astype(str) == "aligned"]
    if aligned.empty:
        return aligned.copy()
    ordered = aligned.copy()
    ordered["_sort_bar"] = pd.to_datetime(ordered["effective_bar_timestamp"], errors="coerce")
    ordered["_sort_chosen"] = pd.to_datetime(ordered["chosen_timestamp"], errors="coerce")
    ordered["_sort_id"] = ordered["event_id"].astype(str)
    ordered = ordered.sort_values(
        ["cluster_id", "_sort_bar", "_sort_chosen", "_sort_id"],
        kind="mergesort",
    )
    chosen = ordered.groupby("cluster_id", as_index=False, sort=False).first()
    return chosen.drop(columns=["_sort_bar", "_sort_chosen", "_sort_id"])


def build_pead_event(
    event: dict,
    index: MarketIndex,
    impact_row: Optional[dict] = None,
) -> dict:
    reasons: list[str] = []
    impact_row = impact_row or {}
    result = {
        "event_id": event.get("event_id"),
        "symbol": event.get("symbol"),
        "cluster_id": event.get("cluster_id"),
        "cluster_size": event.get("cluster_size"),
        "chosen_timestamp": event.get("chosen_timestamp"),
        "effective_bar_timestamp": event.get("effective_bar_timestamp"),
        "trading_date": event.get("effective_trading_date") or event.get("trading_date"),
        "is_financial_result": is_true_flag(event.get("is_financial_result")),
    }

    session_close = pd.to_numeric(impact_row.get("return_session_close"), errors="coerce")
    result["return_session_close"] = float(session_close) if pd.notna(session_close) else np.nan
    result["initial_reaction"] = classify_initial_reaction(result["return_session_close"])
    if pd.isna(result["return_session_close"]):
        reasons.append("missing_session_close")

    anchor_ts = pd.to_datetime(event.get("effective_bar_timestamp"), errors="coerce")
    if pd.isna(anchor_ts):
        reasons.append("excluded_event")
        return _empty_pead(result, reasons)

    first_offset = next(iter(PEAD_HORIZONS.values()))
    _, _, _, market_anchor = future_session_return(index, anchor_ts, first_offset)
    stage_anchor = pd.to_numeric(impact_row.get("anchor_close"), errors="coerce")
    result["anchor_close"] = float(stage_anchor) if pd.notna(stage_anchor) else market_anchor
    if pd.isna(result["anchor_close"]):
        reasons.append("excluded_event")
        return _empty_pead(result, reasons)

    for name, offset in PEAD_HORIZONS.items():
        value, usable, reason, _ = future_session_return(index, anchor_ts, offset)
        result[f"return_{name}"] = value
        result[f"usable_{name}"] = usable
        if not usable and reason:
            reasons.append(reason)

    result["missing_reason"] = ";".join(dict.fromkeys(reasons))
    return result


def _empty_pead(result: dict, reasons: list[str]) -> dict:
    if "anchor_close" not in result:
        result["anchor_close"] = np.nan
    for name in PEAD_HORIZONS:
        result[f"return_{name}"] = np.nan
        result[f"usable_{name}"] = False
    result["missing_reason"] = ";".join(dict.fromkeys(reasons))
    return result


def summarize_group(returns: pd.Series) -> dict[str, object]:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    n = int(len(values))
    row = {
        "n": n,
        "mean_return": float(values.mean()) if n else np.nan,
        "median_return": float(values.median()) if n else np.nan,
        "std_return": float(values.std(ddof=1)) if n > 1 else np.nan,
        "standard_error": np.nan,
        "ci95_lower": np.nan,
        "ci95_upper": np.nan,
        "sample_note": "descriptive" if n < DESCRIPTIVE_N else "inferential",
    }
    if n >= 2:
        mean, low, high, error = mean_confidence_interval(values)
        se = float(values.std(ddof=1) / np.sqrt(n))
        row["standard_error"] = se
        row["ci95_lower"] = low
        row["ci95_upper"] = high
        if error is None and mean is not None:
            row["mean_return"] = mean
    return row


def summarize_pead(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    horizons = list(PEAD_HORIZONS)
    reactions = ["all", "positive", "neutral", "negative"]

    def add_rows(scope: str, symbol: str, subset: pd.DataFrame) -> None:
        for reaction in reactions:
            part = subset if reaction == "all" else subset[subset["initial_reaction"] == reaction]
            for horizon in horizons:
                usable = part[part[f"usable_{horizon}"] == True]  # noqa: E712
                row = {
                    "scope": scope,
                    "symbol": symbol,
                    "initial_reaction": reaction,
                    "horizon": horizon,
                }
                row.update(summarize_group(usable[f"return_{horizon}"]))
                rows.append(row)

    add_rows("pooled", "ALL", events)
    for symbol, stock in events.groupby("symbol", dropna=False):
        add_rows("stock", symbol, stock)
    return pd.DataFrame(rows)


def build_pead_audit(clusters: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    financial = clusters[clusters["is_financial_result"].map(is_true_flag)]
    rows = [
        ("financial_result_announcements", int(len(financial))),
        ("independent_financial_result_events", int(len(events))),
        ("positive_initial_reaction", int((events["initial_reaction"] == "positive").sum())),
        ("neutral_initial_reaction", int((events["initial_reaction"] == "neutral").sum())),
        ("negative_initial_reaction", int((events["initial_reaction"] == "negative").sum())),
        ("unavailable_initial_reaction", int((events["initial_reaction"] == "unavailable").sum())),
    ]
    for name in PEAD_HORIZONS:
        usable = int(events[f"usable_{name}"].sum())
        rows.append((f"usable_{name}", usable))
        rows.append((f"missing_{name}", int(len(events) - usable)))
    return pd.DataFrame(rows, columns=["metric", "value"])


def has_surprise_fields(frame: pd.DataFrame) -> bool:
    lowered = [str(col).lower() for col in frame.columns]
    return any(any(token in col for token in SURPRISE_FIELD_TOKENS) for col in lowered)


def impact_lookup(impact: pd.DataFrame) -> dict[str, dict]:
    if impact.empty:
        return {}
    return {str(row["event_id"]): row for row in impact.to_dict(orient="records")}
