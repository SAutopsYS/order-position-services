"""Event-level price, volume, and range impact.

Returns use close-to-close from the aligned event bar:

    future_close / anchor_close - 1

Intraday 5m/30m/60m stay inside the event session. If the session ends
before the horizon, the metric is missing. Session close uses the last
actual bar of that session.

D+n uses inferred trading sessions, not calendar days. D+1 is the next
session last-bar close over the event-bar close.

Volume ratio:

    event_window_volume / median(same clock-time window in prior sessions)

Look back at most 20 prior sessions. Require at least 5 usable prior
windows. Future sessions are never used.

Primary range metric:

    window_high / window_low - 1

Invalid OHLC bars are not repaired. A metric that needs such a bar is
marked unavailable. Zero-volume bars stay in price calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data_quality import invalid_ohlc_mask
from src.timestamps import Session, infer_sessions

INTRADAY_MINUTES = {"5m": 5, "30m": 30, "60m": 60}
SESSION_HORIZON = "session_close"
MULTI_DAY = {"d1": 1, "d5": 5, "d10": 10, "d20": 20}
BASELINE_LOOKBACK_SESSIONS = 20
MIN_BASELINE_SESSIONS = 5
CI_Z = 1.96


@dataclass
class MarketIndex:
    symbol: str
    frame: pd.DataFrame
    sessions: list[Session]
    bars_by_date: dict
    session_by_date: dict


def build_market_index(market: pd.DataFrame, symbol: str) -> MarketIndex:
    frame = market.sort_values("timestamp").reset_index(drop=True).copy()
    frame["invalid_ohlc"] = invalid_ohlc_mask(frame)
    frame["session_date"] = frame["timestamp"].dt.date
    frame["clock"] = frame["timestamp"].dt.time
    sessions = infer_sessions(frame)
    bars_by_date = {
        trading_date: group.copy()
        for trading_date, group in frame.groupby("session_date", sort=True)
    }
    session_by_date = {session.trading_date: session for session in sessions}
    return MarketIndex(
        symbol=symbol,
        frame=frame,
        sessions=sessions,
        bars_by_date=bars_by_date,
        session_by_date=session_by_date,
    )


def mean_confidence_interval(
    values: list[float] | np.ndarray,
    z: float = CI_Z,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """Normal-approximation 95% CI for a mean. Needs at least 2 values."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return None, None, None, "insufficient_sample"
    mean = float(np.mean(clean))
    se = float(np.std(clean, ddof=1) / np.sqrt(clean.size))
    return mean, mean - z * se, mean + z * se, None


def _session_for_date(index: MarketIndex, trading_date) -> Optional[Session]:
    return index.session_by_date.get(trading_date)


def _session_bars(index: MarketIndex, session: Session) -> pd.DataFrame:
    bars = index.bars_by_date.get(session.trading_date)
    if bars is None:
        return index.frame.iloc[0:0].copy()
    return bars


def _bar_at_or_after(session_bars: pd.DataFrame, target: pd.Timestamp) -> Optional[pd.Series]:
    later = session_bars[session_bars["timestamp"] >= target]
    if later.empty:
        return None
    return later.iloc[0]


def _last_bar(session_bars: pd.DataFrame) -> Optional[pd.Series]:
    if session_bars.empty:
        return None
    return session_bars.iloc[-1]


def _window(session_bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return session_bars[
        (session_bars["timestamp"] >= start) & (session_bars["timestamp"] <= end)
    ]


def _price_ok(bar: Optional[pd.Series]) -> bool:
    return bar is not None and not bool(bar["invalid_ohlc"]) and pd.notna(bar["close"]) and bar["close"] != 0


def future_session_return(
    index: MarketIndex,
    anchor_ts: pd.Timestamp,
    offset: int,
) -> tuple[float, bool, Optional[str], float]:
    """Close-to-close return from the event bar to a later inferred session.

    offset is a trading-session count, not a calendar-day count.
    """
    anchor_ts = pd.Timestamp(anchor_ts)
    session = _session_for_date(index, anchor_ts.date())
    if session is None:
        return np.nan, False, "excluded_event", np.nan
    session_bars = _session_bars(index, session)
    match = session_bars[session_bars["timestamp"] == anchor_ts]
    if match.empty:
        return np.nan, False, "excluded_event", np.nan
    anchor_bar = match.iloc[0]
    anchor_close = float(anchor_bar["close"]) if pd.notna(anchor_bar["close"]) else np.nan
    if not _price_ok(anchor_bar):
        return np.nan, False, "invalid_ohlc", anchor_close
    target_index = session.index + offset
    if target_index >= len(index.sessions):
        return np.nan, False, "future_session_unavailable", anchor_close
    last = _last_bar(_session_bars(index, index.sessions[target_index]))
    if last is None:
        return np.nan, False, "future_session_unavailable", anchor_close
    if not _price_ok(last):
        return np.nan, False, "invalid_ohlc", anchor_close
    return float(last["close"] / anchor_bar["close"] - 1), True, None, anchor_close


def compute_event_metrics(event: dict, index: MarketIndex) -> dict:
    reasons: list[str] = []
    result: dict[str, object] = {
        "event_id": event.get("event_id"),
        "symbol": event.get("symbol"),
        "cluster_id": event.get("cluster_id"),
        "cluster_size": event.get("cluster_size"),
        "subject_group": event.get("subject_group"),
        "is_financial_result": event.get("is_financial_result"),
        "chosen_timestamp": event.get("chosen_timestamp"),
        "effective_bar_timestamp": event.get("effective_bar_timestamp"),
        "trading_date": event.get("effective_trading_date") or event.get("trading_date"),
    }

    if str(event.get("alignment_status", "")) != "aligned":
        reasons.append("excluded_event")
        return _empty_metrics(result, reasons)

    anchor_ts = pd.to_datetime(event.get("effective_bar_timestamp"), errors="coerce")
    if pd.isna(anchor_ts):
        reasons.append("excluded_event")
        return _empty_metrics(result, reasons)

    trading_date = pd.Timestamp(anchor_ts).date()
    session = _session_for_date(index, trading_date)
    if session is None:
        reasons.append("excluded_event")
        return _empty_metrics(result, reasons)

    session_bars = _session_bars(index, session)
    anchor = session_bars[session_bars["timestamp"] == anchor_ts]
    if anchor.empty:
        reasons.append("excluded_event")
        return _empty_metrics(result, reasons)
    anchor_bar = anchor.iloc[0]
    if not _price_ok(anchor_bar):
        reasons.append("invalid_ohlc")
        return _empty_metrics(result, reasons, anchor_bar)

    result.update(_anchor_fields(anchor_bar))
    prior_sessions = [s for s in index.sessions if s.index < session.index]

    for name, minutes in INTRADAY_MINUTES.items():
        _fill_intraday(result, reasons, session_bars, anchor_bar, minutes, name, prior_sessions, index)

    _fill_session_close(result, reasons, session_bars, anchor_bar, prior_sessions, index)
    _fill_multi_day(result, reasons, index, session, anchor_bar)
    result["missing_reason"] = ";".join(dict.fromkeys(reasons))
    return result


def _anchor_fields(bar: pd.Series) -> dict[str, object]:
    return {
        "anchor_open": float(bar["open"]),
        "anchor_high": float(bar["high"]),
        "anchor_low": float(bar["low"]),
        "anchor_close": float(bar["close"]),
        "anchor_volume": float(bar["volume"]),
    }


def _empty_metrics(result: dict, reasons: list[str], anchor: Optional[pd.Series] = None) -> dict:
    if anchor is not None:
        result.update(_anchor_fields(anchor))
    else:
        for key in ("anchor_open", "anchor_high", "anchor_low", "anchor_close", "anchor_volume"):
            result[key] = np.nan
    for name in ("5m", "30m", "60m", "session_close", "d1", "d5", "d10", "d20"):
        result[f"return_{name}"] = np.nan
        result[f"usable_{name}"] = False
    for name in ("5m", "30m", "60m", "session_close"):
        result[f"volume_ratio_{name}"] = np.nan
        result[f"range_{name}"] = np.nan
    result["missing_reason"] = ";".join(dict.fromkeys(reasons))
    return result


def _fill_intraday(
    result: dict,
    reasons: list[str],
    session_bars: pd.DataFrame,
    anchor_bar: pd.Series,
    minutes: int,
    name: str,
    prior_sessions: list[Session],
    index: MarketIndex,
) -> None:
    target = pd.Timestamp(anchor_bar["timestamp"]) + timedelta(minutes=minutes)
    horizon_bar = _bar_at_or_after(session_bars, target)
    if horizon_bar is None:
        result[f"return_{name}"] = np.nan
        result[f"volume_ratio_{name}"] = np.nan
        result[f"range_{name}"] = np.nan
        result[f"usable_{name}"] = False
        reasons.append("session_ended_before_horizon")
        return
    _fill_window_metrics(
        result, reasons, session_bars, anchor_bar, horizon_bar, name, prior_sessions, index
    )


def _fill_session_close(
    result: dict,
    reasons: list[str],
    session_bars: pd.DataFrame,
    anchor_bar: pd.Series,
    prior_sessions: list[Session],
    index: MarketIndex,
) -> None:
    last = _last_bar(session_bars)
    if last is None:
        result["return_session_close"] = np.nan
        result["volume_ratio_session_close"] = np.nan
        result["range_session_close"] = np.nan
        result["usable_session_close"] = False
        reasons.append("session_ended_before_horizon")
        return
    _fill_window_metrics(
        result, reasons, session_bars, anchor_bar, last, "session_close", prior_sessions, index
    )


def _fill_window_metrics(
    result: dict,
    reasons: list[str],
    session_bars: pd.DataFrame,
    anchor_bar: pd.Series,
    end_bar: pd.Series,
    name: str,
    prior_sessions: list[Session],
    index: MarketIndex,
) -> None:
    if not _price_ok(end_bar):
        result[f"return_{name}"] = np.nan
        result[f"volume_ratio_{name}"] = np.nan
        result[f"range_{name}"] = np.nan
        result[f"usable_{name}"] = False
        reasons.append("invalid_ohlc")
        return

    window = _window(session_bars, anchor_bar["timestamp"], end_bar["timestamp"])
    if window.empty or window["invalid_ohlc"].any():
        result[f"return_{name}"] = np.nan
        result[f"volume_ratio_{name}"] = np.nan
        result[f"range_{name}"] = np.nan
        result[f"usable_{name}"] = False
        reasons.append("invalid_ohlc" if window["invalid_ohlc"].any() else "insufficient_future_bars")
        return

    result[f"return_{name}"] = float(end_bar["close"] / anchor_bar["close"] - 1)
    result[f"range_{name}"] = float(window["high"].max() / window["low"].min() - 1)
    event_volume = float(window["volume"].sum())
    expected = _expected_volume(index, prior_sessions, anchor_bar["timestamp"], end_bar["timestamp"])
    if expected is None:
        result[f"volume_ratio_{name}"] = np.nan
        reasons.append("insufficient_baseline")
    elif expected == 0:
        result[f"volume_ratio_{name}"] = np.nan
        reasons.append("missing_volume")
    else:
        result[f"volume_ratio_{name}"] = event_volume / expected
    result[f"usable_{name}"] = True


def _clock(ts: pd.Timestamp) -> time:
    return pd.Timestamp(ts).time()


def _expected_volume(
    index: MarketIndex,
    prior_sessions: list[Session],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Optional[float]:
    start_clock = _clock(start_ts)
    end_clock = _clock(end_ts)
    values: list[float] = []
    lookback = prior_sessions[-BASELINE_LOOKBACK_SESSIONS:]
    for session in lookback:
        bars = _session_bars(index, session)
        window = bars[(bars["clock"] >= start_clock) & (bars["clock"] <= end_clock)]
        if window.empty:
            continue
        values.append(float(window["volume"].sum()))
    if len(values) < MIN_BASELINE_SESSIONS:
        return None
    return float(np.median(values))


def _fill_multi_day(
    result: dict,
    reasons: list[str],
    index: MarketIndex,
    event_session: Session,
    anchor_bar: pd.Series,
) -> None:
    for name, offset in MULTI_DAY.items():
        target_index = event_session.index + offset
        if target_index >= len(index.sessions):
            result[f"return_{name}"] = np.nan
            result[f"usable_{name}"] = False
            reasons.append("future_session_unavailable")
            continue
        target = index.sessions[target_index]
        last = _last_bar(_session_bars(index, target))
        if last is None or not _price_ok(last):
            result[f"return_{name}"] = np.nan
            result[f"usable_{name}"] = False
            reasons.append("invalid_ohlc" if last is not None else "future_session_unavailable")
            continue
        result[f"return_{name}"] = float(last["close"] / anchor_bar["close"] - 1)
        result[f"usable_{name}"] = True


def summarize_impacts(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    horizons = [
        ("5m", "return_5m", "volume_ratio_5m", "range_5m", "usable_5m"),
        ("30m", "return_30m", "volume_ratio_30m", "range_30m", "usable_30m"),
        ("60m", "return_60m", "volume_ratio_60m", "range_60m", "usable_60m"),
        ("session_close", "return_session_close", "volume_ratio_session_close", "range_session_close", "usable_session_close"),
        ("d1", "return_d1", None, None, "usable_d1"),
        ("d5", "return_d5", None, None, "usable_d5"),
        ("d10", "return_d10", None, None, "usable_d10"),
        ("d20", "return_d20", None, None, "usable_d20"),
    ]
    for (symbol, group), part in results.groupby(["symbol", "subject_group"], dropna=False):
        for horizon, ret_col, vol_col, range_col, usable_col in horizons:
            usable = part[part[usable_col] == True]  # noqa: E712
            rets = pd.to_numeric(usable[ret_col], errors="coerce").dropna()
            row = {
                "symbol": symbol,
                "subject_group": group,
                "horizon": horizon,
                "n": int(len(rets)),
                "mean_return": float(rets.mean()) if len(rets) else np.nan,
                "median_return": float(rets.median()) if len(rets) else np.nan,
                "std_return": float(rets.std(ddof=1)) if len(rets) > 1 else np.nan,
                "mean_volume_ratio": np.nan,
                "median_volume_ratio": np.nan,
                "mean_range": np.nan,
                "median_range": np.nan,
            }
            if vol_col is not None:
                vols = pd.to_numeric(usable[vol_col], errors="coerce").dropna()
                row["mean_volume_ratio"] = float(vols.mean()) if len(vols) else np.nan
                row["median_volume_ratio"] = float(vols.median()) if len(vols) else np.nan
            if range_col is not None:
                ranges = pd.to_numeric(usable[range_col], errors="coerce").dropna()
                row["mean_range"] = float(ranges.mean()) if len(ranges) else np.nan
                row["median_range"] = float(ranges.median()) if len(ranges) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)
