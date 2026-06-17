"""Lexicon sentiment analyzer, classification bands, and DB backfill."""

import pytest

from market_intel.search import score_sentiment_pending
from market_intel.sentiment import (
    NEGATIVE,
    NEGATORS,
    POSITIVE,
    LexiconSentiment,
    classify,
    get_default_analyzer,
)
from market_intel.storage.db import init_db, make_engine, make_session_factory
from market_intel.storage.models import NewsArticle
from market_intel.storage.news_repo import upsert_articles


@pytest.fixture
def analyzer():
    return LexiconSentiment()


def test_polarity_direction(analyzer):
    pos, neg, neutral = analyzer.score(
        [
            "Stocks surge as profits beat estimates and growth accelerates",
            "Shares plunge on recession fears and mounting losses",
            "Company schedules its annual general meeting for Tuesday",
        ]
    )
    assert pos > 0
    assert neg < 0
    assert neutral == 0.0


def test_empty_and_no_polarity_are_zero(analyzer):
    assert analyzer.score(["", "the quick brown fox", None]) == [0.0, 0.0, 0.0]


def test_negation_flips_polarity(analyzer):
    # "not ... strong" should not read as positive.
    assert analyzer.score(["not a strong quarter for the firm"])[0] <= 0
    # "without losses" should not read as negative.
    assert analyzer.score(["a clean year without losses"])[0] >= 0


def test_scores_bounded_and_length_matches(analyzer):
    texts = ["gain gain gain", "loss loss", "surge then plunge", "neutral wording here"]
    scores = analyzer.score(texts)
    assert len(scores) == len(texts)
    assert all(-1.0 <= s <= 1.0 for s in scores)
    # balanced positive/negative cancels out
    assert analyzer.score(["surge then plunge"])[0] == 0.0


def test_default_analyzer_is_lexicon():
    a = get_default_analyzer()
    assert a.name == "lexicon"
    assert a.score(["record profit and strong growth"])[0] > 0


def test_lexicons_are_disjoint():
    assert not (POSITIVE & NEGATIVE)
    assert not (POSITIVE & NEGATORS)
    assert not (NEGATIVE & NEGATORS)


def test_classify_bands():
    assert classify(0.5) == "positive"
    assert classify(-0.5) == "negative"
    assert classify(0.0) == "neutral"
    assert classify(0.05) == "neutral"  # inclusive neutral band edge
    assert classify(None) == "unknown"


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/sent.db")
    init_db(engine)
    return make_session_factory(engine)


def test_score_sentiment_pending_scores_only_nulls_and_is_idempotent(session_factory):
    articles = [
        {"url_hash": "h1", "url": "u1", "title": "Profits surge on record growth"},
        {"url_hash": "h2", "url": "u2", "title": "Losses mount amid recession fears"},
    ]
    with session_factory() as s:
        upsert_articles(s, articles)

    with session_factory() as s:
        assert score_sentiment_pending(s) == 2  # both were NULL
    with session_factory() as s:
        assert score_sentiment_pending(s) == 0  # nothing left to score
        rows = {a.url_hash: a.sentiment for a in s.query(NewsArticle).all()}
        assert rows["h1"] > 0
        assert rows["h2"] < 0


def test_score_sentiment_pending_respects_limit(session_factory):
    articles = [{"url_hash": f"h{i}", "url": f"u{i}", "title": "growth"} for i in range(5)]
    with session_factory() as s:
        upsert_articles(s, articles)
    with session_factory() as s:
        assert score_sentiment_pending(s, limit=2) == 2
    with session_factory() as s:
        scored = s.query(NewsArticle).filter(NewsArticle.sentiment.is_not(None)).count()
        assert scored == 2
