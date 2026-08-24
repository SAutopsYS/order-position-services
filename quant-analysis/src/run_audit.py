"""Run the Stage 1 data audit and write machine-readable outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.config import MARKET_FILENAMES, parse_path_args
from src.data_loader import load_all_markets, load_announcements, market_bar_times
from src.data_quality import (
    duplicate_timestamp_mask,
    format_optional_timestamp,
    invalid_ohlc_mask,
    negative_volume_mask,
    summarize_issues,
    unexpected_minute_gaps,
    zero_volume_mask,
)
from src.timestamps import (
    align_announcement_to_bar,
    infer_sessions,
    market_timestamps_are_sorted,
    parse_announcement_timestamp,
)


def build_alignment_table(
    announcements: pd.DataFrame,
    markets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    bar_times = {symbol: market_bar_times(frame) for symbol, frame in markets.items()}
    sessions = {
        symbol: infer_sessions(frame) for symbol, frame in markets.items()
    }
    rows: list[dict[str, object]] = []
    for record in announcements.to_dict(orient="records"):
        parsed = parse_announcement_timestamp(record)
        symbol = record.get("symbol")
        base = {
            "announcement_id": record.get("NEWSID"),
            "symbol": symbol,
            "original_DissemDT": record.get("DissemDT"),
            "original_DT_TM": record.get("DT_TM"),
            "chosen_timestamp": format_optional_timestamp(parsed.timestamp),
            "timestamp_source": parsed.source or "",
            "effective_event_timestamp": format_optional_timestamp(parsed.timestamp),
            "effective_trading_date": "",
            "effective_bar_timestamp": "",
            "alignment_status": "",
            "exclusion_reason": parsed.error or "",
        }
        if parsed.timestamp is None:
            base["alignment_status"] = "excluded"
            rows.append(base)
            continue
        if symbol not in bar_times:
            base["alignment_status"] = "excluded"
            base["exclusion_reason"] = "unknown symbol"
            rows.append(base)
            continue
        aligned = align_announcement_to_bar(
            parsed.timestamp, bar_times[symbol], sessions[symbol]
        )
        base["effective_trading_date"] = (
            "" if aligned.effective_trading_date is None else str(aligned.effective_trading_date)
        )
        base["effective_bar_timestamp"] = format_optional_timestamp(
            aligned.effective_bar_timestamp
        )
        base["alignment_status"] = aligned.status
        base["exclusion_reason"] = aligned.exclusion_reason or ""
        rows.append(base)
    return pd.DataFrame(rows)


def audit_announcements(announcements: pd.DataFrame) -> list[dict[str, object]]:
    issues = [
        summarize_issues("announcements", "row_count", len(announcements)),
        summarize_issues("announcements", "column_count", announcements.shape[1]),
        summarize_issues(
            "announcements",
            "duplicate_newsid",
            int(announcements["NEWSID"].duplicated().sum()),
        ),
        summarize_issues(
            "announcements",
            "duplicate_rows",
            int(announcements.duplicated().sum()),
        ),
        summarize_issues(
            "announcements",
            "unmapped_scrip",
            int(announcements["symbol"].isna().sum()),
        ),
    ]
    for column, missing in announcements.isna().sum().items():
        if missing:
            issues.append(
                summarize_issues("announcements", f"missing_{column}", int(missing))
            )
    return issues


def audit_market(symbol: str, market: pd.DataFrame) -> list[dict[str, object]]:
    issues = [
        summarize_issues(symbol, "row_count", len(market)),
        summarize_issues(symbol, "unique_sessions", market["timestamp"].dt.date.nunique()),
        summarize_issues(
            symbol,
            "unsorted_timestamps",
            0 if market_timestamps_are_sorted(market["timestamp"]) else 1,
        ),
        summarize_issues(
            symbol,
            "duplicate_timestamps",
            int(duplicate_timestamp_mask(market).sum() // 2),
        ),
        summarize_issues(symbol, "invalid_ohlc", int(invalid_ohlc_mask(market).sum())),
        summarize_issues(symbol, "negative_volume", int(negative_volume_mask(market).sum())),
        summarize_issues(symbol, "zero_volume", int(zero_volume_mask(market).sum())),
        summarize_issues(
            symbol,
            "intraday_gaps_gt_1min",
            len(unexpected_minute_gaps(market)),
        ),
    ]
    return issues


def run_audit(data_dir: str | Path | None = None, output_dir: str | Path | None = None) -> dict:
    if data_dir is None or output_dir is None:
        paths = parse_path_args(None if data_dir is None else ["--data-dir", str(data_dir)])
    else:
        from src.config import resolve_paths

        paths = resolve_paths(data_dir, output_dir)

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    announcements = load_announcements(paths.announcements_csv)
    markets = load_all_markets(paths)
    alignment = build_alignment_table(announcements, markets)

    quality_rows: list[dict[str, object]] = []
    quality_rows.extend(audit_announcements(announcements))
    for symbol, market in markets.items():
        quality_rows.extend(audit_market(symbol, market))

    fallback_count = int((alignment["timestamp_source"] == "DT_TM").sum())
    excluded = alignment[alignment["alignment_status"] == "excluded"]
    quality_rows.append(
        summarize_issues("alignment", "timestamp_fallback_dt_tm", fallback_count)
    )
    quality_rows.append(
        summarize_issues("alignment", "exclusions", len(excluded))
    )
    quality_rows.append(
        summarize_issues("alignment", "aligned", int((alignment["alignment_status"] == "aligned").sum()))
    )

    quality = pd.DataFrame(quality_rows)
    quality_path = paths.output_dir / "data_quality_summary.csv"
    alignment_path = paths.output_dir / "event_alignment_audit.csv"
    quality.to_csv(quality_path, index=False)
    alignment.to_csv(alignment_path, index=False)

    print("Announcements", len(announcements))
    print("Symbols", sorted(announcements["symbol"].dropna().unique().tolist()))
    print("Timestamp fallback DT_TM", fallback_count)
    print("Alignment excluded", len(excluded))
    print("Alignment aligned", int((alignment["alignment_status"] == "aligned").sum()))
    for symbol, filename in MARKET_FILENAMES.items():
        market = markets[symbol]
        print(
            f"{symbol} rows={len(market)} sessions={market['timestamp'].dt.date.nunique()} "
            f"invalid_ohlc={int(invalid_ohlc_mask(market).sum())} "
            f"zero_volume={int(zero_volume_mask(market).sum())}"
        )
    print("Wrote", quality_path)
    print("Wrote", alignment_path)
    return {
        "announcements": len(announcements),
        "fallback_count": fallback_count,
        "excluded": len(excluded),
        "quality_path": quality_path,
        "alignment_path": alignment_path,
    }


def main(argv: list[str] | None = None) -> int:
    paths = parse_path_args(argv)
    run_audit(paths.data_dir, paths.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
