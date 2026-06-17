"""API endpoints via FastAPI TestClient against a seeded SQLite DB."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from market_intel.api.app import _num, create_app
from market_intel.search import embed_pending, score_sentiment_pending
from market_intel.storage.db import init_db, make_engine, make_session_factory
from market_intel.storage.filings_repo import upsert_filings
from market_intel.storage.macro_repo import upsert_macro
from market_intel.storage.news_repo import upsert_articles
from market_intel.storage.prices_repo import upsert_prices


@pytest.fixture
def client(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/api.db")
    init_db(engine)
    sf = make_session_factory(engine)

    idx = pd.date_range("2020-01-02", periods=5, freq="B")
    prices = pd.DataFrame(
        {
            "Open": [10, 11, 12, 13, 14.0],
            "High": [11, 12, 13, 14, 15.0],
            "Low": [9, 10, 11, 12, 13.0],
            "Close": [10.5, 11.5, 12.5, 13.5, 14.5],
            "Volume": [100, 110, 120, 130, 140.0],
        },
        index=idx,
    )
    macro = pd.DataFrame(
        {"value": [100.0, 101.0, 102.0]}, index=pd.date_range("2020-01-01", periods=3, freq="MS")
    )
    articles = [
        {"url_hash": "h1", "url": "u1", "title": "Oil supply disruption at remote port"},
        {"url_hash": "h2", "url": "u2", "title": "Shares plunge on losses and recession fears"},
    ]
    filings = [{"accession_no": "a-1", "cik": "0000320193", "ticker": "AAPL", "form": "10-K"}]
    with sf() as s:
        upsert_prices(s, prices, "AAPL")
        upsert_macro(s, macro, "GDP")
        upsert_articles(s, articles)
        upsert_filings(s, filings)
        embed_pending(s)
        score_sentiment_pending(s)
    return TestClient(create_app(session_factory=sf))


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_prices(client):
    rows = client.get("/api/prices/AAPL").json()
    assert len(rows) == 5
    assert rows[0]["close"] == 10.5
    assert set(rows[0]) == {"date", "open", "high", "low", "close", "volume"}


def test_prices_limit_and_case_insensitive(client):
    assert len(client.get("/api/prices/aapl?limit=2").json()) == 2


def test_prices_unknown_symbol_empty(client):
    assert client.get("/api/prices/ZZZZ").json() == []


def test_indicators(client):
    rows = client.get("/api/indicators/aapl").json()
    assert len(rows) == 5  # one row per price bar
    assert set(rows[0]) == {
        "date",
        "sma_20",
        "sma_50",
        "ema_20",
        "bb_upper",
        "bb_mid",
        "bb_lower",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
    }
    # Only 5 bars: every window-20 column stays in warm-up (null everywhere),
    # so they must serialize as JSON null; EMA/RSI/MACD are defined from row 0/1.
    for col in ("sma_20", "sma_50", "bb_upper", "bb_mid", "bb_lower"):
        assert all(r[col] is None for r in rows)
    assert rows[-1]["ema_20"] is not None
    assert rows[-1]["rsi_14"] is not None
    assert rows[-1]["macd"] is not None


def test_indicators_limit_and_unknown_symbol(client):
    assert len(client.get("/api/indicators/AAPL?limit=2").json()) == 2
    assert client.get("/api/indicators/ZZZZ").json() == []


def test_num_coerces_non_finite_to_none():
    # Infinity is not valid JSON; non-finite values must serialize as null.
    assert _num(float("inf")) is None
    assert _num(float("-inf")) is None
    assert _num(float("nan")) is None
    assert _num(None) is None
    assert _num(1.5) == 1.5


def test_macro(client):
    rows = client.get("/api/macro/GDP").json()
    assert [r["value"] for r in rows] == [100.0, 101.0, 102.0]


def test_news_recent(client):
    rows = client.get("/api/news/recent").json()
    assert len(rows) == 2
    assert "title" in rows[0] and "seen_date" in rows[0]
    assert {"sentiment", "sentiment_label"} <= rows[0].keys()


def test_news_sentiment_labels(client):
    by_url = {r["url"]: r for r in client.get("/api/news/recent").json()}
    # "Bank shares plunge on mounting losses and recession fears" -> clearly negative
    assert by_url["u2"]["sentiment_label"] == "negative"
    assert by_url["u2"]["sentiment"] < 0
    # The oil headline carries no lexicon polarity -> neutral, score 0.0
    assert by_url["u1"]["sentiment_label"] == "neutral"
    assert by_url["u1"]["sentiment"] == 0.0


def test_news_search_includes_sentiment(client):
    rows = client.get("/api/news/search?q=plunge").json()
    assert rows and rows[0]["sentiment_label"] == "negative"


def test_news_search_keyword(client):
    rows = client.get("/api/news/search?q=oil").json()
    assert len(rows) == 1
    assert "Oil" in rows[0]["title"]


def test_news_search_semantic(client):
    rows = client.get("/api/news/search?q=oil supply shock&semantic=true").json()
    assert rows
    assert "score" in rows[0]
    assert rows[0]["url"] == "u1"  # most relevant first


def test_filings(client):
    rows = client.get("/api/filings/AAPL").json()
    assert rows[0]["form"] == "10-K"
    assert rows[0]["accession_no"] == "a-1"


def test_index_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "MARKET INTELLIGENCE TERMINAL" in resp.text
