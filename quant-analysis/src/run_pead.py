"""Run Stage 4 price-conditioned PEAD from Stage 2/3 outputs and market files."""

from __future__ import annotations

import sys

import pandas as pd

from src.config import parse_path_args, resolve_paths
from src.data_loader import load_all_markets
from src.event_impact import build_market_index
from src.pead import (
    PEAD_HORIZONS,
    build_pead_audit,
    build_pead_event,
    impact_lookup,
    is_true_flag,
    select_independent_financial_events,
    summarize_pead,
)


def _print_summary_block(title: str, summary: pd.DataFrame, scope: str) -> None:
    print(title, flush=True)
    part = summary[summary["scope"] == scope]
    cols = [
        "symbol",
        "initial_reaction",
        "horizon",
        "n",
        "mean_return",
        "median_return",
        "standard_error",
        "ci95_lower",
        "ci95_upper",
        "sample_note",
    ]
    print(part[cols].to_string(index=False), flush=True)


def run_pead(data_dir=None, output_dir=None) -> dict:
    if data_dir is None:
        paths = parse_path_args()
    else:
        paths = resolve_paths(data_dir, output_dir or "outputs")
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    clusters_path = paths.output_dir / "event_clusters.csv"
    impact_path = paths.output_dir / "event_level_results.csv"
    if not clusters_path.is_file():
        raise FileNotFoundError(f"Missing {clusters_path}. Run Stage 2 before Stage 4.")
    if not impact_path.is_file():
        raise FileNotFoundError(f"Missing {impact_path}. Run Stage 3 before Stage 4.")

    clusters = pd.read_csv(clusters_path)
    impact = pd.read_csv(impact_path)
    reps = select_independent_financial_events(clusters)
    print("Financial-result announcements", int(clusters["is_financial_result"].map(is_true_flag).sum()), flush=True)
    print("Independent financial-result events", len(reps), flush=True)

    markets = load_all_markets(paths)
    indexes = {
        symbol: build_market_index(frame, symbol) for symbol, frame in markets.items()
    }
    lookup = impact_lookup(impact)

    rows: list[dict] = []
    for record in reps.to_dict(orient="records"):
        symbol = record.get("symbol")
        if symbol not in indexes:
            continue
        rows.append(build_pead_event(record, indexes[symbol], lookup.get(str(record.get("event_id")))))

    events = pd.DataFrame(rows)
    summary = summarize_pead(events)
    audit = build_pead_audit(clusters, events)

    events_path = paths.output_dir / "pead_event_results.csv"
    summary_path = paths.output_dir / "pead_summary.csv"
    audit_path = paths.output_dir / "pead_audit.csv"
    events.to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)
    audit.to_csv(audit_path, index=False)

    print("positive", int((events["initial_reaction"] == "positive").sum()), flush=True)
    print("neutral", int((events["initial_reaction"] == "neutral").sum()), flush=True)
    print("negative", int((events["initial_reaction"] == "negative").sum()), flush=True)
    for name in PEAD_HORIZONS:
        usable = int(events[f"usable_{name}"].sum())
        print(f"usable_{name}", usable, "missing", len(events) - usable, flush=True)
    _print_summary_block("Pooled PEAD descriptive statistics", summary, "pooled")
    _print_summary_block("Stock-wise PEAD descriptive statistics", summary, "stock")
    print("Wrote", events_path, flush=True)
    print("Wrote", summary_path, flush=True)
    print("Wrote", audit_path, flush=True)
    return {"events": events, "summary": summary, "audit": audit}


def main(argv: list[str] | None = None) -> int:
    paths = parse_path_args(argv)
    run_pead(paths.data_dir, paths.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
