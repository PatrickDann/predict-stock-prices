"""News search: embedding backfill, keyword (LIKE), semantic (cosine) on SQLite."""

import pytest

from market_intel.search import embed_pending, keyword_search, semantic_search
from market_intel.storage.db import init_db, make_engine, make_session_factory
from market_intel.storage.news_repo import upsert_articles

ARTICLES = [
    {"url_hash": "h1", "url": "u1", "title": "Oil supply disruption at remote port"},
    {"url_hash": "h2", "url": "u2", "title": "Central bank raises interest rates sharply"},
    {"url_hash": "h3", "url": "u3", "title": "Local football club wins championship"},
]


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/test.db")
    init_db(engine)
    sf = make_session_factory(engine)
    with sf() as s:
        upsert_articles(s, ARTICLES)
    return sf


def test_embed_pending_then_idempotent(session_factory):
    with session_factory() as s:
        n = embed_pending(s)
    assert n == 3
    with session_factory() as s:
        assert embed_pending(s) == 0  # nothing left to embed


def test_keyword_search(session_factory):
    with session_factory() as s:
        hits = keyword_search(s, "oil")
        assert [a.url for a in hits] == ["u1"]
        # case-insensitive
        assert keyword_search(s, "BANK")[0].url == "u2"
        assert keyword_search(s, "nonexistentword") == []


def test_semantic_search_ranks_relevant_first(session_factory):
    with session_factory() as s:
        embed_pending(s)
    with session_factory() as s:
        results = semantic_search(s, "oil supply shock at the port", limit=3)
    assert len(results) == 3
    # most lexically-similar article ranks first; scores are descending
    assert results[0][0].url == "u1"
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_semantic_search_skips_unembedded(session_factory):
    # no embed_pending call -> no embeddings -> no semantic results
    with session_factory() as s:
        assert semantic_search(s, "oil") == []
