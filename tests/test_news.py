"""GDELT parsing, fetch (stubbed), storage, and end-to-end ingest."""

import pytest

from market_intel.ingest.news import (
    fetch_gdelt_articles,
    ingest_gdelt,
    parse_gdelt_articles,
)
from market_intel.storage.db import init_db, make_engine, make_session_factory
from market_intel.storage.news_repo import (
    count_articles,
    country_counts,
    get_recent_articles,
    upsert_articles,
)

PAYLOAD = {
    "articles": [
        {
            "url": "https://example.com/a",
            "title": "Oil supply shock in remote port",
            "seendate": "20240115T123000Z",
            "domain": "example.com",
            "language": "English",
            "sourcecountry": "United States",
        },
        {  # duplicate url -> deduped within batch
            "url": "https://example.com/a",
            "title": "Oil supply shock in remote port (dup)",
            "seendate": "20240115T130000Z",
        },
        {  # missing title -> dropped by validation gate
            "url": "https://example.com/b",
            "seendate": "20240115T140000Z",
        },
        {
            "url": "https://example.org/c",
            "title": "Central bank surprise",
            "seendate": "bad-date",  # unparseable -> seen_date None
        },
    ]
}


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/test.db")
    init_db(engine)
    return make_session_factory(engine)


def test_parse_dedups_and_validates():
    arts = parse_gdelt_articles(PAYLOAD)
    urls = {a["url"] for a in arts}
    assert urls == {"https://example.com/a", "https://example.org/c"}  # b dropped, a deduped
    a = next(x for x in arts if x["url"] == "https://example.com/a")
    assert a["seen_date"] is not None and a["source_country"] == "United States"
    assert len(a["url_hash"]) == 64
    c = next(x for x in arts if x["url"] == "https://example.org/c")
    assert c["seen_date"] is None  # unparseable date tolerated


def test_parse_empty():
    assert parse_gdelt_articles({}) == []
    assert parse_gdelt_articles({"articles": None}) == []


def test_fetch_builds_request():
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return PAYLOAD

    def fake_get(url, params=None, timeout=None):
        captured["params"] = params
        return FakeResp()

    out = fetch_gdelt_articles("oil", max_records=10, timespan="3d", get=fake_get)
    assert out == PAYLOAD
    assert captured["params"]["query"] == "oil"
    assert captured["params"]["mode"] == "ArtList"
    assert captured["params"]["format"] == "json"
    assert captured["params"]["maxrecords"] == 10
    assert captured["params"]["timespan"] == "3d"


def test_ingest_gdelt_end_to_end_and_idempotent(session_factory):
    def fake_get(*a, **k):
        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return PAYLOAD

        return R()

    with session_factory() as s:
        n = ingest_gdelt(s, "oil", get=fake_get)
    assert n == 2  # a (deduped) + c
    with session_factory() as s:
        ingest_gdelt(s, "oil", get=fake_get)  # re-ingest
    with session_factory() as s:
        assert count_articles(s) == 2  # no duplicates
        recent = get_recent_articles(s, limit=10)
        assert len(recent) == 2
        # the article with a real seen_date sorts ahead of the NULL one
        assert recent[0].url == "https://example.com/a"


def test_country_counts_aggregates_and_excludes_blank(session_factory):
    articles = [
        {"url_hash": "1", "url": "u1", "title": "t1", "source_country": "Egypt"},
        {"url_hash": "2", "url": "u2", "title": "t2", "source_country": "Egypt"},
        {"url_hash": "3", "url": "u3", "title": "t3", "source_country": "India"},
        {"url_hash": "4", "url": "u4", "title": "t4", "source_country": None},
        {"url_hash": "5", "url": "u5", "title": "t5", "source_country": ""},
    ]
    with session_factory() as s:
        upsert_articles(s, articles)
        counts = country_counts(s)
    assert counts == [("Egypt", 2), ("India", 1)]  # most active first; NULL/"" dropped
