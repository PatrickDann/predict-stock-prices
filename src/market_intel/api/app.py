"""FastAPI application factory and JSON routes."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from market_intel.indicators import compute_indicators
from market_intel.search import keyword_search, semantic_search
from market_intel.storage.db import init_db, make_engine, make_session_factory
from market_intel.storage.filings_repo import get_filings
from market_intel.storage.macro_repo import get_macro
from market_intel.storage.news_repo import get_recent_articles
from market_intel.storage.prices_repo import get_prices

STATIC_DIR = Path(__file__).parent / "static"


def _num(value) -> float | None:
    """Coerce to a JSON-safe number (NaN -> None)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _price_records(df: pd.DataFrame, limit: int) -> list[dict]:
    if limit:
        df = df.tail(limit)
    return [
        {
            "date": pd.Timestamp(ts).date().isoformat(),
            "open": _num(row.get("Open")),
            "high": _num(row.get("High")),
            "low": _num(row.get("Low")),
            "close": _num(row.get("Close")),
            "volume": _num(row.get("Volume")),
        }
        for ts, row in df.iterrows()
    ]


def _indicator_records(df: pd.DataFrame, limit: int) -> list[dict]:
    """Date-aligned indicator rows with JSON-safe (NaN -> None) values."""
    if limit:
        df = df.tail(limit)
    return [
        {"date": pd.Timestamp(ts).date().isoformat(), **{c: _num(row[c]) for c in df.columns}}
        for ts, row in df.iterrows()
    ]


def _article_record(article, score: float | None = None) -> dict:
    record = {
        "title": article.title,
        "url": article.url,
        "domain": article.domain,
        "language": article.language,
        "source_country": article.source_country,
        "seen_date": article.seen_date.isoformat() if article.seen_date else None,
    }
    if score is not None:
        record["score"] = round(float(score), 4)
    return record


def _default_session_factory():
    engine = make_engine()
    init_db(engine)
    return make_session_factory(engine)


def create_app(session_factory=None) -> FastAPI:
    """Build the API. Pass ``session_factory`` (e.g. a SQLite one) for tests."""
    sf = session_factory or _default_session_factory()
    app = FastAPI(title="Market Intelligence Terminal", version="0.1.0")

    def get_session() -> Iterator[Session]:
        with sf() as session:
            yield session

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/prices/{symbol}")
    def prices(
        symbol: str,
        limit: int = Query(500, ge=0, le=10000),
        session: Session = Depends(get_session),
    ) -> list[dict]:
        return _price_records(get_prices(session, symbol.upper()), limit)

    @app.get("/api/indicators/{symbol}")
    def indicators(
        symbol: str,
        limit: int = Query(500, ge=0, le=10000),
        session: Session = Depends(get_session),
    ) -> list[dict]:
        # Compute over the full series so warm-up NaNs don't eat the window,
        # then trim to the trailing `limit` rows for display. An unknown symbol
        # yields an empty frame -> empty indicator frame -> [].
        df = get_prices(session, symbol.upper())
        return _indicator_records(compute_indicators(df), limit)

    @app.get("/api/macro/{series_id}")
    def macro(
        series_id: str,
        limit: int = Query(2000, ge=0, le=20000),
        session: Session = Depends(get_session),
    ) -> list[dict]:
        df = get_macro(session, series_id.upper())
        if limit:
            df = df.tail(limit)
        return [
            {"date": pd.Timestamp(ts).date().isoformat(), "value": _num(row["value"])}
            for ts, row in df.iterrows()
        ]

    @app.get("/api/news/recent")
    def news_recent(
        limit: int = Query(50, ge=1, le=500),
        session: Session = Depends(get_session),
    ) -> list[dict]:
        return [_article_record(a) for a in get_recent_articles(session, limit=limit)]

    @app.get("/api/news/search")
    def news_search(
        q: str,
        semantic: bool = False,
        limit: int = Query(20, ge=1, le=200),
        session: Session = Depends(get_session),
    ) -> list[dict]:
        if semantic:
            return [_article_record(a, s) for a, s in semantic_search(session, q, limit=limit)]
        return [_article_record(a) for a in keyword_search(session, q, limit=limit)]

    @app.get("/api/filings/{ticker}")
    def filings(
        ticker: str,
        limit: int = Query(50, ge=1, le=500),
        session: Session = Depends(get_session),
    ) -> list[dict]:
        return [
            {
                "form": f.form,
                "filing_date": f.filing_date.isoformat() if f.filing_date else None,
                "accession_no": f.accession_no,
                "primary_doc": f.primary_doc,
                "cik": f.cik,
            }
            for f in get_filings(session, ticker, limit=limit)
        ]

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
