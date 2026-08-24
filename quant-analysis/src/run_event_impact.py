"""Compute Stage 3 event-level impact from clustered events and market files."""

from __future__ import annotations

import sys

import pandas as pd

from src.config import parse_path_args, resolve_paths
from src.data_loader import load_all_markets
from src.event_impact import build_market_index, compute_event_metrics, summarize_impacts


def run_event_impact(data_dir=None, output_dir=None) -> dict:
    if data_dir is None:
        paths = parse_path_args()
    else:
        paths = resolve_paths(data_dir, output_dir or "outputs")
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    events_path = paths.output_dir / "event_clusters.csv"
    if not events_path.is_file():
        raise FileNotFoundError(
            f"Missing {events_path}. Run Stage 2 taxonomy before Stage 3."
        )
    events = pd.read_csv(events_path)
    print("Loaded", len(events), "event rows", flush=True)
    markets = load_all_markets(paths)
    indexes = {
        symbol: build_market_index(frame, symbol) for symbol, frame in markets.items()
    }
    print("Indexed", len(indexes), "symbols", flush=True)

    rows: list[dict] = []
    for record in events.to_dict(orient="records"):
        symbol = record.get("symbol")
        if symbol not in indexes:
            continue
        rows.append(compute_event_metrics(record, indexes[symbol]))

    results = pd.DataFrame(rows)
    summary = summarize_impacts(results)
    results_path = paths.output_dir / "event_level_results.csv"
    summary_path = paths.output_dir / "event_impact_summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    aligned = events[events["alignment_status"] == "aligned"]
    print("Event records", len(results))
    print("Aligned source events", len(aligned))
    for horizon in ("5m", "30m", "60m", "session_close", "d1", "d5", "d10", "d20"):
        usable = int(results[f"usable_{horizon}"].sum())
        print(f"usable_{horizon}", usable, "missing", len(results) - usable)
    invalid = results["missing_reason"].fillna("").str.contains("invalid_ohlc")
    baseline = results["missing_reason"].fillna("").str.contains("insufficient_baseline")
    print("events with invalid_ohlc reason", int(invalid.sum()))
    print("events with insufficient_baseline reason", int(baseline.sum()))
    print("Wrote", results_path)
    print("Wrote", summary_path)
    return {"results": results, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    paths = parse_path_args(argv)
    run_event_impact(paths.data_dir, paths.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
