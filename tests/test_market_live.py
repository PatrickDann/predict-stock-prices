"""Live yfinance normalization + ingestion (downloader injected, no network)."""

import pandas as pd
import pytest

from market_intel.ingest.market import ingest_yfinance, normalize_yf
from market_intel.storage.db import init_db, make_engine, make_session_factory
from market_intel.storage.prices_repo import get_prices


def _flat_df():
    idx = pd.date_range("2020-01-02", periods=3, freq="B")
    return pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "High": [11.0, 12.0, 13.0],
            "Low": [9.0, 10.0, 11.0],
            "Close": [10.5, 11.5, 12.5],
            "Volume": [1000, 1100, 1200],
        },
        index=idx,
    )


def _multiindex_df():
    df = _flat_df()
    df.columns = pd.MultiIndex.from_product([list(df.columns), ["AAPL"]])
    return df


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/test.db")
    init_db(engine)
    return make_session_factory(engine)


def test_normalize_flat():
    out = normalize_yf(_flat_df(), "AAPL")
    assert list(out.columns) == ["Close", "High", "Low", "Open", "Volume"]
    assert out["Close"].tolist() == [10.5, 11.5, 12.5]


def test_normalize_multiindex():
    out = normalize_yf(_multiindex_df(), "AAPL")
    assert list(out.columns) == ["Close", "High", "Low", "Open", "Volume"]
    assert out["Close"].tolist() == [10.5, 11.5, 12.5]


def test_ingest_yfinance_with_injected_downloader(session_factory):
    def fake_download(ticker, start=None, end=None, auto_adjust=True, progress=False):
        return _multiindex_df()

    with session_factory() as s:
        n = ingest_yfinance(s, "AAPL", downloader=fake_download)
    assert n == 3
    with session_factory() as s:
        assert get_prices(s, "AAPL")["Close"].tolist() == [10.5, 11.5, 12.5]


def test_ingest_yfinance_empty(session_factory):
    with session_factory() as s:
        assert ingest_yfinance(s, "AAPL", downloader=lambda *a, **k: pd.DataFrame()) == 0


def test_normalize_lowercase_ticker_multiindex():
    # caller passes 'aapl' but columns store 'AAPL' — must not KeyError
    out = normalize_yf(_multiindex_df(), "aapl")
    assert out["Close"].tolist() == [10.5, 11.5, 12.5]


def test_normalize_reversed_multiindex():
    # (ticker, field) order instead of (field, ticker)
    df = _flat_df()
    df.columns = pd.MultiIndex.from_product([["AAPL"], list(df.columns)])
    out = normalize_yf(df, "AAPL")
    assert list(out.columns) == ["Close", "High", "Low", "Open", "Volume"]
    assert out["Close"].tolist() == [10.5, 11.5, 12.5]


def test_normalize_tz_aware_index_keeps_trading_day():
    df = _flat_df()
    df.index = df.index.tz_localize("America/New_York")
    out = normalize_yf(df, "AAPL")
    assert out.index.tz is None  # tz dropped
    # the calendar dates are unchanged (no off-by-one)
    assert [d.strftime("%Y-%m-%d") for d in out.index] == ["2020-01-02", "2020-01-03", "2020-01-06"]


def test_normalize_drops_nan_close():
    df = _flat_df()
    df.loc[df.index[1], "Close"] = float("nan")
    out = normalize_yf(df, "AAPL")
    assert len(out) == 2  # the NaN-Close row dropped


def test_ingest_yfinance_all_nan_close(session_factory):
    df = _flat_df()
    df["Close"] = float("nan")
    with session_factory() as s:
        assert ingest_yfinance(s, "AAPL", downloader=lambda *a, **k: df) == 0
