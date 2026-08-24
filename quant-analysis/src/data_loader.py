"""Load supplied announcement and market files without changing them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MARKET_FILENAMES, SCRIP_TO_SYMBOL, Paths


def load_announcements(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Announcement file not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    frame["symbol"] = frame["SCRIP_CD"].map(SCRIP_TO_SYMBOL)
    return frame


def load_market(path: Path, symbol: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Market file not found: {path}")
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["date"], format="ISO8601", errors="coerce")
    frame["symbol"] = symbol
    return frame


def load_all_markets(paths: Paths) -> dict[str, pd.DataFrame]:
    markets: dict[str, pd.DataFrame] = {}
    for symbol, filename in MARKET_FILENAMES.items():
        markets[symbol] = load_market(paths.market_dir / filename, symbol)
    return markets


def market_bar_times(market: pd.DataFrame) -> np.ndarray:
    timestamps = market["timestamp"].dropna().sort_values()
    return timestamps.to_numpy(dtype="datetime64[ns]")
