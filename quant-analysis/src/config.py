"""Configurable paths and symbol mapping for the Quant assessment data."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

# Scrip codes from the supplied metadata.json.
SCRIP_TO_SYMBOL = {
    500325: "RELIANCE",
    500180: "HDFCBANK",
    543384: "NYKAA",
    541154: "HAL",
    542649: "RVNL",
}

SYMBOL_TO_SCRIP = {symbol: scrip for scrip, symbol in SCRIP_TO_SYMBOL.items()}

MARKET_FILENAMES = {
    "RELIANCE": "RELIANCE.csv",
    "HDFCBANK": "HDFCBANK.csv",
    "NYKAA": "NYKAA.csv",
    "HAL": "HAL.csv",
    "RVNL": "RVNL.csv",
}

ANNOUNCEMENTS_CSV_NAME = "corporate_announcements.csv"
ANNOUNCEMENTS_JSONL_NAME = "corporate_announcements.jsonl"


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    announcements_csv: Path
    announcements_jsonl: Path
    market_dir: Path
    output_dir: Path


def resolve_paths(
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Paths:
    """Resolve data and output paths from arguments or environment variables."""
    raw_data = data_dir or os.environ.get("QUANT_DATA_DIR")
    if not raw_data:
        raise ValueError(
            "Data directory is required. Pass --data-dir or set QUANT_DATA_DIR."
        )
    data_path = Path(raw_data)
    out_path = Path(output_dir or os.environ.get("QUANT_OUTPUT_DIR", "outputs"))
    return Paths(
        data_dir=data_path,
        announcements_csv=data_path / ANNOUNCEMENTS_CSV_NAME,
        announcements_jsonl=data_path / ANNOUNCEMENTS_JSONL_NAME,
        market_dir=data_path,
        output_dir=out_path,
    )


def parse_path_args(argv: list[str] | None = None) -> Paths:
    parser = argparse.ArgumentParser(description="Quant analysis data paths")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("QUANT_DATA_DIR"),
        help="Directory that contains the supplied assessment CSVs",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("QUANT_OUTPUT_DIR", "outputs"),
        help="Directory for generated audit files (default: outputs)",
    )
    args = parser.parse_args(argv)
    return resolve_paths(args.data_dir, args.output_dir)
