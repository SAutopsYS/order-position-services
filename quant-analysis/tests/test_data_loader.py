from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_announcements, load_market
from src.data_quality import (
    duplicate_timestamp_mask,
    invalid_ohlc_mask,
    zero_volume_mask,
)
from src.timestamps import market_timestamps_are_sorted


def test_market_timestamps_remain_sorted(tmp_path: Path) -> None:
    path = tmp_path / "RELIANCE.csv"
    path.write_text(
        "date,open,high,low,close,volume,oi\n"
        "2024-01-02T09:15:00.000000,10,11,9,10,100,0\n"
        "2024-01-02T09:16:00.000000,10,11,9,10,100,0\n",
        encoding="utf-8",
    )
    market = load_market(path, "RELIANCE")
    assert market_timestamps_are_sorted(market["timestamp"]) is True


def test_unsorted_market_timestamps_are_detected(tmp_path: Path) -> None:
    path = tmp_path / "RELIANCE.csv"
    path.write_text(
        "date,open,high,low,close,volume,oi\n"
        "2024-01-02T09:16:00.000000,10,11,9,10,100,0\n"
        "2024-01-02T09:15:00.000000,10,11,9,10,100,0\n",
        encoding="utf-8",
    )
    market = load_market(path, "RELIANCE")
    assert market_timestamps_are_sorted(market["timestamp"]) is False


def test_invalid_ohlc_rows_are_detected() -> None:
    market = pd.DataFrame(
        {
            "open": [10.0, 10.0],
            "high": [11.0, 9.5],
            "low": [9.0, 9.0],
            "close": [10.5, 10.0],
            "volume": [100.0, 100.0],
        }
    )
    mask = invalid_ohlc_mask(market)
    assert mask.tolist() == [False, True]


def test_duplicate_timestamps_are_detected() -> None:
    market = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02T09:15:00", "2024-01-02T09:15:00", "2024-01-02T09:16:00"]
            )
        }
    )
    mask = duplicate_timestamp_mask(market)
    assert int(mask.sum()) == 2


def test_zero_volume_is_recorded_not_repaired() -> None:
    market = pd.DataFrame({"volume": [100.0, 0.0, 50.0]})
    mask = zero_volume_mask(market)
    assert mask.tolist() == [False, True, False]
    assert market.loc[mask, "volume"].tolist() == [0.0]


def test_load_announcements_maps_scrip(tmp_path: Path) -> None:
    path = tmp_path / "corporate_announcements.csv"
    path.write_text(
        "NEWSID,SCRIP_CD,DissemDT,DT_TM\n"
        "evt-1,500325,2024-01-02T09:15:00,2024-01-02T09:15:00\n",
        encoding="utf-8",
    )
    frame = load_announcements(path)
    assert frame.loc[0, "symbol"] == "RELIANCE"


def test_missing_announcement_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_announcements(tmp_path / "missing.csv")
