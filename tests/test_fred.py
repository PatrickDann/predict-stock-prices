"""FRED parsing, fetch (stubbed), validation, and end-to-end ingest."""

import pandas as pd
import pytest

from market_intel.ingest.macro import (
    fetch_fred_observations,
    ingest_macro_frame,
    parse_fred_observations,
)
from market_intel.ingest.validation import validate_macro
from market_intel.storage.db import init_db, make_engine, make_session_factory
from market_intel.storage.macro_repo import get_macro

PAYLOAD = {
    "observations": [
        {"date": "2020-03-01", "value": "102.5"},
        {"date": "2020-01-01", "value": "100.0"},  # out of order -> sorted
        {"date": "2020-02-01", "value": "."},  # FRED missing marker -> dropped
    ]
}


def test_parse_drops_missing_and_sorts():
    df = parse_fred_observations(PAYLOAD)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert df["value"].tolist() == [100.0, 102.5]  # "." row gone, sorted by date


def test_parse_empty_payload():
    df = parse_fred_observations({"observations": []})
    assert df.empty


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

    out = fetch_fred_observations("GDP", "SECRET", get=fake_get)
    assert out == PAYLOAD
    assert captured["params"] == {
        "series_id": "GDP",
        "api_key": "SECRET",
        "file_type": "json",
    }


def test_fetch_raises_on_http_error():
    class FakeResp:
        def raise_for_status(self):
            raise RuntimeError("400")

        def json(self):  # pragma: no cover
            return {}

    with pytest.raises(RuntimeError):
        fetch_fred_observations("GDP", "K", get=lambda *a, **k: FakeResp())


def test_validate_macro_rejects_non_datetime_index():
    df = pd.DataFrame({"value": [1.0, 2.0]})  # RangeIndex
    with pytest.raises(ValueError):
        validate_macro(df)


def test_ingest_fred_frame_end_to_end(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/t.db")
    init_db(engine)
    sf = make_session_factory(engine)
    df = parse_fred_observations(PAYLOAD)
    with sf() as s:
        n = ingest_macro_frame(s, df, "GDP")
    assert n == 2
    with sf() as s:
        assert get_macro(s, "GDP")["value"].tolist() == [100.0, 102.5]
