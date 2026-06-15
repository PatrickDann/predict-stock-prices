"""Loader handles both single- and multi-ticker yfinance CSV shapes."""

import pandas as pd
import pytest

from market_intel.data.loaders import load_price_frame, load_prices

DATA_DIR = "data"
FIELDS = ["Close", "High", "Low", "Open", "Volume"]


def test_load_single_ticker():
    df = load_prices("AAPL", data_dir=DATA_DIR)
    assert list(df.columns) == FIELDS
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert df["Close"].isna().sum() == 0
    assert len(df) > 2000


def test_load_ticker_from_multi_ticker_file():
    msft = load_prices("MSFT", data_dir=DATA_DIR, dataset="tech")
    aapl = load_prices("AAPL", data_dir=DATA_DIR, dataset="tech")
    assert list(msft.columns) == FIELDS
    assert msft["Close"].isna().sum() == 0
    # different tickers -> different series
    assert not msft["Close"].equals(aapl["Close"])


def test_multi_ticker_frame_exposes_all_tickers():
    frame = load_price_frame(f"{DATA_DIR}/tech_stock_data.csv")
    tickers = set(frame.columns.get_level_values(1))
    assert {"AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA"} <= tickers


def test_unknown_ticker_raises():
    with pytest.raises(KeyError):
        load_prices("NOPE", data_dir=DATA_DIR, dataset="tech")


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_prices("AAPL", data_dir="data", dataset="does_not_exist")


# --- synthetic CSVs (no dependency on committed data) ---


def test_csv_without_index_name_row_keeps_all_rows(tmp_path):
    """The old hardcoded skiprows=[2] would silently drop the first data row."""
    (tmp_path / "xyz_stock_data.csv").write_text(
        "Price,Close,High,Low,Open,Volume\n"
        "Ticker,XYZ,XYZ,XYZ,XYZ,XYZ\n"
        "2020-01-02,10,11,9,10,1000\n"
        "2020-01-03,12,13,11,12,1100\n"
    )
    df = load_prices("XYZ", data_dir=tmp_path)
    assert len(df) == 2  # neither real row dropped
    assert df["Close"].tolist() == [10.0, 12.0]


def test_csv_with_index_row_whitespace_and_nan(tmp_path):
    (tmp_path / "abc_stock_data.csv").write_text(
        "Price ,Close ,High ,Low ,Open ,Volume \n"  # trailing whitespace in headers
        "Ticker,ABC,ABC,ABC,ABC,ABC\n"
        "Date,,,,,\n"  # spurious index-name row -> dropped
        "2020-01-06,12,13,11,12,1100\n"
        "2020-01-03,10,11,9,10,1000\n"  # out of order -> sorted
        "2020-01-07,,,,,\n"  # NaN-close row -> dropped by load_prices
    )
    frame = load_price_frame(tmp_path / "abc_stock_data.csv")
    assert ("Close", "ABC") in frame.columns  # whitespace stripped

    df = load_prices("abc", data_dir=tmp_path)  # case-insensitive ticker
    assert len(df) == 2  # Date row + NaN row dropped
    assert df.index.is_monotonic_increasing
    assert df["Close"].tolist() == [10.0, 12.0]


def test_empty_after_parse_raises(tmp_path):
    (tmp_path / "empty_stock_data.csv").write_text(
        "Price,Close,High,Low,Open,Volume\nTicker,E,E,E,E,E\nDate,,,,,\n"
    )
    with pytest.raises(ValueError):
        load_price_frame(tmp_path / "empty_stock_data.csv")
