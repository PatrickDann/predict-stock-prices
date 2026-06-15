"""End-to-end: CSV -> validate -> DB, idempotent."""

import pandas as pd
import pytest

from market_intel.ingest.market import ingest_from_csv, ingest_price_frame
from market_intel.storage.db import init_db, make_engine, make_session_factory
from market_intel.storage.prices_repo import get_prices


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/test.db")
    init_db(engine)
    return make_session_factory(engine)


def _write_csv(path):
    path.write_text(
        "Price,Close,High,Low,Open,Volume\n"
        "Ticker,XYZ,XYZ,XYZ,XYZ,XYZ\n"
        "Date,,,,,\n"
        "2020-01-02,10,11,9,10,1000\n"
        "2020-01-03,12,13,11,12,1100\n"
    )


def test_ingest_from_csv(tmp_path, session_factory):
    _write_csv(tmp_path / "xyz_stock_data.csv")
    with session_factory() as s:
        n = ingest_from_csv(s, "xyz", data_dir=tmp_path)
    assert n == 2
    with session_factory() as s:
        out = get_prices(s, "XYZ")  # symbol upper-cased on ingest
    assert out["Close"].tolist() == [10.0, 12.0]


def test_ingest_from_csv_idempotent(tmp_path, session_factory):
    _write_csv(tmp_path / "xyz_stock_data.csv")
    with session_factory() as s:
        ingest_from_csv(s, "XYZ", data_dir=tmp_path)
    with session_factory() as s:
        ingest_from_csv(s, "XYZ", data_dir=tmp_path)  # again
    with session_factory() as s:
        assert len(get_prices(s, "XYZ")) == 2  # no duplicates


def test_ingest_rejects_bad_frame(session_factory):
    idx = pd.date_range("2020-01-02", periods=2, freq="B")
    bad = pd.DataFrame(
        {
            "Open": [10.0, 10.0],
            "High": [11.0, 11.0],
            "Low": [9.0, 9.0],
            "Close": [10.0, -5.0],
            "Volume": [1.0, 1.0],
        },
        index=idx,
    )
    from pandera.errors import SchemaError, SchemaErrors

    with session_factory() as s:
        with pytest.raises((SchemaError, SchemaErrors)):
            ingest_price_frame(s, bad, "BAD")
