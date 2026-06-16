"""FastAPI application factory and JSON routes."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from market_intel.cache import Cache, build_cache, cached
from market_intel.config import settings
from market_intel.indicators import compute_indicators
from market_intel.search import keyword_search, semantic_search
from market_intel.storage.db import init_db, make_engine, make_session_factory
from market_intel.storage.filings_repo import get_filings
from market_intel.storage.macro_repo import get_macro
from market_intel.storage.news_repo import get_recent_articles, latest_seen
from market_intel.storage.prices_repo import get_prices

STATIC_DIR = Path(__file__).parent / "static"

CACHE_PREFIX = "mi:v1"  # bump if a cached payload's shape changes
STREAM_NEWS_LIMIT = 50  # articles pushed per SSE news event

# SSE headers: never cache the stream; disable proxy buffering so frames flush at once.
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_UNSET = object()  # sentinel so the stream emits news on its very first tick


def _sse(event: str, data: object) -> str:
    """Format one Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _cache_key(resource: str, *parts: object) -> str:
    """Build a versioned read-cache key (bump ``CACHE_PREFIX`` if a payload shape changes)."""
    return ":".join([CACHE_PREFIX, resource, *(str(p) for p in parts)])


def _num(value) -> float | None:
    """Coerce to a JSON-safe number (NaN/±inf -> None).

    ``Infinity`` is not valid JSON, so non-finite values become ``None``.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


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


def _news_snapshot(sf, last_seen: object, limit: int) -> tuple[object, list[dict] | None]:
    """Cheap change check (runs off the event loop): return (latest_seen, records|None).

    ``records`` is the recent-article list on the first call or whenever the latest
    timestamp changed, else ``None`` (nothing new to push).
    """
    with sf() as s:
        seen = latest_seen(s)
        if last_seen is _UNSET or seen != last_seen:
            return seen, [_article_record(a) for a in get_recent_articles(s, limit=limit)]
        return seen, None


async def news_stream(
    sf,
    poll: float,
    is_disconnected: Callable[[], bool | Awaitable[bool]],
    *,
    news_limit: int = STREAM_NEWS_LIMIT,
) -> AsyncIterator[str]:
    """Yield SSE frames until ``is_disconnected()`` is truthy.

    Emits a ``news`` event on the first cycle and whenever fresh articles land, plus a
    ``tick`` every cycle (doubles as the keepalive heartbeat). ``is_disconnected`` may be
    sync or async (FastAPI's ``Request.is_disconnected`` is a coroutine). The DB read runs
    in a thread so the sync session never blocks the event loop.
    """
    last_seen: object = _UNSET
    while True:
        disconnected = is_disconnected()
        if inspect.isawaitable(disconnected):
            disconnected = await disconnected
        if disconnected:
            break
        seen, records = await asyncio.to_thread(_news_snapshot, sf, last_seen, news_limit)
        if records is not None:
            last_seen = seen
            yield _sse("news", records)
        yield _sse("tick", {"ts": int(time.time())})
        await asyncio.sleep(poll)


def create_app(
    session_factory=None,
    cache: Cache | None = None,
    stream_poll_seconds: float | None = None,
) -> FastAPI:
    """Build the API.

    Pass ``session_factory`` (e.g. a SQLite one) and ``cache`` (e.g. a ``MemoryCache``)
    for tests; ``stream_poll_seconds`` overrides the SSE poll interval (also for tests).
    """
    sf = session_factory or _default_session_factory()
    cache = cache if cache is not None else build_cache(enabled=settings.cache_enabled)
    ttl = settings.cache_ttl_seconds
    poll = settings.stream_poll_seconds if stream_poll_seconds is None else stream_poll_seconds
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
        symbol = symbol.upper()
        return cached(
            cache,
            _cache_key("prices", symbol, limit),
            ttl,
            lambda: _price_records(get_prices(session, symbol), limit),
        )

    @app.get("/api/indicators/{symbol}")
    def indicators(
        symbol: str,
        limit: int = Query(500, ge=0, le=10000),
        session: Session = Depends(get_session),
    ) -> list[dict]:
        # Compute over the full series so warm-up NaNs don't eat the window,
        # then trim to the trailing `limit` rows for display. An unknown symbol
        # yields an empty frame -> empty indicator frame -> [].
        symbol = symbol.upper()
        return cached(
            cache,
            _cache_key("indicators", symbol, limit),
            ttl,
            lambda: _indicator_records(compute_indicators(get_prices(session, symbol)), limit),
        )

    @app.get("/api/macro/{series_id}")
    def macro(
        series_id: str,
        limit: int = Query(2000, ge=0, le=20000),
        session: Session = Depends(get_session),
    ) -> list[dict]:
        series_id = series_id.upper()

        def produce() -> list[dict]:
            df = get_macro(session, series_id)
            if limit:
                df = df.tail(limit)
            return [
                {"date": pd.Timestamp(ts).date().isoformat(), "value": _num(row["value"])}
                for ts, row in df.iterrows()
            ]

        return cached(cache, _cache_key("macro", series_id, limit), ttl, produce)

    @app.get("/api/news/recent")
    def news_recent(
        limit: int = Query(50, ge=1, le=500),
        session: Session = Depends(get_session),
    ) -> list[dict]:
        return cached(
            cache,
            _cache_key("news:recent", limit),
            ttl,
            lambda: [_article_record(a) for a in get_recent_articles(session, limit=limit)],
        )

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
        ticker = ticker.upper()
        return cached(
            cache,
            _cache_key("filings", ticker, limit),
            ttl,
            lambda: [
                {
                    "form": f.form,
                    "filing_date": f.filing_date.isoformat() if f.filing_date else None,
                    "accession_no": f.accession_no,
                    "primary_doc": f.primary_doc,
                    "cik": f.cik,
                }
                for f in get_filings(session, ticker, limit=limit)
            ],
        )

    @app.get("/api/stream")
    async def stream(request: Request) -> StreamingResponse:
        """Server-Sent Events: push the news feed as fresh articles land, plus a periodic
        ``tick`` the client uses to refresh the symbol panels. EventSource auto-reconnects.

        The stream polls the DB (the worker is a separate process) and emits ``news`` only
        when the latest article timestamp changes — pub/sub fan-out is a later upgrade.
        """
        return StreamingResponse(
            news_stream(sf, poll, request.is_disconnected),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
