"""SEC EDGAR CIK resolution, filings parse, fetch headers, end-to-end ingest."""

import pytest

from market_intel.ingest.filings import (
    ingest_edgar,
    ingest_filings,
    parse_filings,
    resolve_cik,
)
from market_intel.storage.db import init_db, make_engine, make_session_factory
from market_intel.storage.filings_repo import get_filings

TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}

SUBMISSIONS = {
    "cik": "320193",
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-24-000010", "0000320193-24-000009", ""],
            "form": ["10-K", "8-K", "10-Q"],
            "filingDate": ["2024-11-01", "2024-08-01", "2024-05-01"],
            "primaryDocument": ["aapl-10k.htm", "d8k.htm", "q.htm"],
            "primaryDocDescription": ["10-K", "8-K", ""],
        }
    },
}


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/test.db")
    init_db(engine)
    return make_session_factory(engine)


def test_resolve_cik_and_sets_user_agent():
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return _Resp(TICKERS)

    assert resolve_cik("aapl", get=fake_get, user_agent="me/1.0") == "0000320193"
    assert captured["headers"]["User-Agent"] == "me/1.0"
    assert resolve_cik("NOPE", get=fake_get, user_agent="me/1.0") is None


def test_parse_filings_skips_blank_accession():
    out = parse_filings(SUBMISSIONS, ticker="AAPL")
    assert len(out) == 2  # the blank-accession row is dropped
    assert out[0]["accession_no"] == "0000320193-24-000010"
    assert out[0]["cik"] == "0000320193"
    assert out[0]["ticker"] == "AAPL"
    assert out[0]["form"] == "10-K"
    assert str(out[0]["filing_date"]) == "2024-11-01"


def test_ingest_edgar_end_to_end(session_factory):
    def fake_get(url, headers=None, timeout=None):
        return _Resp(TICKERS if "company_tickers" in url else SUBMISSIONS)

    with session_factory() as s:
        n = ingest_edgar(s, "AAPL", get=fake_get)
    assert n == 2
    with session_factory() as s:
        filings = get_filings(s, "AAPL")
    assert [f.form for f in filings] == ["10-K", "8-K"]  # newest first


def test_ingest_edgar_unknown_ticker_raises(session_factory):
    def fake_get(url, headers=None, timeout=None):
        return _Resp(TICKERS)

    with session_factory() as s:
        with pytest.raises(KeyError):
            ingest_edgar(s, "ZZZZ", get=fake_get)


def test_ingest_filings_idempotent(session_factory):
    from market_intel.storage.filings_repo import get_filings

    with session_factory() as s:
        ingest_filings(s, parse_filings(SUBMISSIONS, "AAPL"))
    # re-ingest with a mutated description -> update, not duplicate
    mutated = parse_filings(SUBMISSIONS, "AAPL")
    mutated[0]["description"] = "amended"
    with session_factory() as s:
        ingest_filings(s, mutated)
    with session_factory() as s:
        filings = get_filings(s, "AAPL")
        assert len(filings) == 2  # no duplicates
        assert filings[0].description == "amended"  # last write wins


def test_parse_drops_whitespace_accession_and_dedups():
    payload = {
        "cik": "1",
        "filings": {
            "recent": {
                "accessionNumber": ["  ", "acc-1", "acc-1"],  # blank + duplicate
                "form": ["10-K", "8-K", "8-K"],
                "filingDate": ["2024-01-01", "2024-02-01", "2024-02-01"],
                "primaryDocument": ["a", "b", "b"],
                "primaryDocDescription": ["", "", ""],
            }
        },
    }
    parsed = parse_filings(payload, "X")
    # whitespace accession dropped; duplicate accession still parsed (PK upsert dedups on write)
    assert [p["accession_no"] for p in parsed] == ["acc-1", "acc-1"]


def test_parse_ragged_arrays_no_crash():
    payload = {
        "cik": "1",
        "filings": {
            "recent": {
                "accessionNumber": ["acc-1", "acc-2"],
                "form": ["10-K"],  # shorter than accessions
                "filingDate": ["2024-01-01"],
                "primaryDocument": [],
                "primaryDocDescription": [],
            }
        },
    }
    parsed = parse_filings(payload, "X")  # must not raise
    assert parsed[0]["accession_no"] == "acc-1" and parsed[0]["form"] == "10-K"
    assert len(parsed) == 1  # acc-2 dropped (no matching form)
