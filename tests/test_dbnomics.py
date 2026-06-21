"""DBnomics parsing, fetch (stubbed), and end-to-end ingest."""

import pandas as pd
import pytest
from sqlalchemy import select

from market_intel.ingest.dbnomics import (
    fetch_dbnomics_series,
    ingest_dbnomics,
    parse_dbnomics_series,
)
from market_intel.storage.db import init_db, make_engine, make_session_factory
from market_intel.storage.macro_repo import get_macro
from market_intel.storage.models import MacroSeries


def _payload(docs):
    return {"series": {"docs": docs}}


# Out of order, with a "NA" missing marker and a clean period_start_day.
PAYLOAD = _payload(
    [
        {
            "period": ["2020-Q2", "2020-Q1", "2020-Q3"],
            "period_start_day": ["2020-04-01", "2020-01-01", "2020-07-01"],
            "value": ["NA", 100.0, 102.5],
        }
    ]
)


def test_parse_drops_na_and_sorts():
    df = parse_dbnomics_series(PAYLOAD)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    # "NA" row gone, sorted by date (uses period_start_day, not the label).
    assert df["value"].tolist() == [100.0, 102.5]
    assert df.index[0] == pd.Timestamp("2020-01-01")


def test_parse_falls_back_to_period_label():
    df = parse_dbnomics_series(_payload([{"period": ["2019", "2020"], "value": [1.0, 2.0]}]))
    assert df.index.tolist() == [pd.Timestamp("2019"), pd.Timestamp("2020")]
    assert df["value"].tolist() == [1.0, 2.0]


def test_parse_truncates_misaligned_arrays():
    # A malformed payload with more dates than values must not raise.
    df = parse_dbnomics_series(
        _payload([{"period_start_day": ["2020-01-01", "2020-02-01"], "value": [5.0]}])
    )
    assert df["value"].tolist() == [5.0]


def test_parse_empty_and_missing_docs():
    assert parse_dbnomics_series(_payload([])).empty
    assert parse_dbnomics_series({}).empty  # no "series" key at all


def test_fetch_builds_correct_request():
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return PAYLOAD

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return FakeResp()

    out = fetch_dbnomics_series("IMF/WEO/USA.NGDP_RPCH", get=fake_get)
    assert out == PAYLOAD
    assert captured["params"] == {
        "series_ids": "IMF/WEO/USA.NGDP_RPCH",
        "observations": 1,
    }


def test_fetch_raises_on_http_error():
    class FakeResp:
        def raise_for_status(self):
            raise RuntimeError("500")

        def json(self):  # pragma: no cover
            return {}

    with pytest.raises(RuntimeError):
        fetch_dbnomics_series("X/Y/Z", get=lambda *a, **k: FakeResp())


def test_ingest_dbnomics_end_to_end_and_idempotent(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/t.db")
    init_db(engine)
    sf = make_session_factory(engine)

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return PAYLOAD

    fake_get = lambda *a, **k: FakeResp()  # noqa: E731
    sid = "Eurostat/namq_10_gdp/Q.CLV10_MEUR.SCA.B1GQ.DE"

    with sf() as s:
        n = ingest_dbnomics(s, sid, get=fake_get)
    assert n == 2

    with sf() as s:
        assert get_macro(s, sid)["value"].tolist() == [100.0, 102.5]
        sources = s.scalars(select(MacroSeries.source).where(MacroSeries.series_id == sid)).all()
        assert set(sources) == {"DBnomics"}

    # Re-ingesting the same series is idempotent (composite PK upsert).
    with sf() as s:
        ingest_dbnomics(s, sid, get=fake_get)
    with sf() as s:
        assert s.query(MacroSeries).filter_by(series_id=sid).count() == 2
