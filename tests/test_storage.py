"""Storage layer against a file-backed SQLite DB (portable with Postgres)."""

import pandas as pd
import pytest

from market_intel.storage.db import init_db, make_engine, make_session_factory
from market_intel.storage.prices_repo import get_prices, upsert_prices


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/test.db")
    init_db(engine)
    return make_session_factory(engine)


def _frame(closes, start="2020-01-02"):
    idx = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Volume": [1000.0] * len(closes),
        },
        index=idx,
    )


def test_upsert_and_read_roundtrip(session_factory):
    df = _frame([10.0, 11.0, 12.0])
    with session_factory() as s:
        n = upsert_prices(s, df, "AAPL")
    assert n == 3

    with session_factory() as s:
        out = get_prices(s, "AAPL")
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out["Close"].tolist() == [10.0, 11.0, 12.0]
    assert out.index.is_monotonic_increasing


def test_upsert_is_idempotent(session_factory):
    df = _frame([10.0, 11.0, 12.0])
    with session_factory() as s:
        upsert_prices(s, df, "AAPL")
    # re-ingest same dates with updated closes -> update, not duplicate
    df2 = _frame([10.0, 11.0, 99.0])
    with session_factory() as s:
        upsert_prices(s, df2, "AAPL")

    with session_factory() as s:
        out = get_prices(s, "AAPL")
    assert len(out) == 3  # no duplicate rows
    assert out["Close"].tolist() == [10.0, 11.0, 99.0]  # last value wins


def test_symbols_isolated(session_factory):
    with session_factory() as s:
        upsert_prices(s, _frame([10.0, 11.0]), "AAPL")
        upsert_prices(s, _frame([20.0, 21.0]), "MSFT")
    with session_factory() as s:
        assert get_prices(s, "AAPL")["Close"].tolist() == [10.0, 11.0]
        assert get_prices(s, "MSFT")["Close"].tolist() == [20.0, 21.0]


def test_nan_volume_stored_as_null(session_factory):
    df = _frame([10.0, 11.0])
    df.loc[df.index[0], "Volume"] = float("nan")
    with session_factory() as s:
        upsert_prices(s, df, "AAPL")
    with session_factory() as s:
        out = get_prices(s, "AAPL")
    assert pd.isna(out["Volume"].iloc[0])
