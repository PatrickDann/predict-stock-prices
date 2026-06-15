"""APScheduler-based ingestion worker.

Jobs are plain functions that open their own DB session from an injected
session factory and swallow/log per-item errors so one failure never kills the
scheduler. ``build_scheduler`` registers the configured jobs and returns the
(unstarted) scheduler, which makes registration unit-testable without running
the blocking loop. FastAPI comes later (Phase 2); ingestion needs only this.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from market_intel.ingest.filings import ingest_edgar
from market_intel.ingest.macro import ingest_fred
from market_intel.ingest.market import ingest_from_csv
from market_intel.ingest.news import ingest_gdelt
from market_intel.search import embed_pending

log = logging.getLogger(__name__)


def ingest_fred_job(series_ids: Sequence[str], api_key: str, session_factory) -> None:
    with session_factory() as session:
        for sid in series_ids:
            try:
                n = ingest_fred(session, sid, api_key)
                log.info("FRED %s: ingested %d rows", sid, n)
            except Exception:
                log.exception("FRED ingest failed for %s", sid)


def ingest_market_csv_job(
    tickers: Sequence[str], session_factory, data_dir: str = "data", dataset: str | None = None
) -> None:
    with session_factory() as session:
        for ticker in tickers:
            try:
                n = ingest_from_csv(session, ticker, data_dir=data_dir, dataset=dataset)
                log.info("market %s: ingested %d rows", ticker, n)
            except Exception:
                log.exception("market ingest failed for %s", ticker)


def ingest_gdelt_job(queries: Sequence[str], session_factory, timespan: str = "1d") -> None:
    with session_factory() as session:
        for query in queries:
            try:
                n = ingest_gdelt(session, query, timespan=timespan)
                log.info("GDELT %r: ingested %d articles", query, n)
            except Exception:
                log.exception("GDELT ingest failed for %r", query)
        try:
            embedded = embed_pending(session)  # keep semantic search up to date
            if embedded:
                log.info("embedded %d new articles", embedded)
        except Exception:
            log.exception("embedding pending articles failed")


def ingest_edgar_job(tickers: Sequence[str], session_factory) -> None:
    with session_factory() as session:
        for ticker in tickers:
            try:
                n = ingest_edgar(session, ticker)
                log.info("EDGAR %s: ingested %d filings", ticker, n)
            except Exception:
                log.exception("EDGAR ingest failed for %s", ticker)


def build_scheduler(
    session_factory,
    *,
    fred_series: Sequence[str] = (),
    fred_api_key: str | None = None,
    market_tickers: Sequence[str] = (),
    market_dataset: str | None = None,
    data_dir: str = "data",
    gdelt_queries: Sequence[str] = (),
    edgar_tickers: Sequence[str] = (),
    scheduler: BlockingScheduler | None = None,
) -> BlockingScheduler:
    """Register configured ingestion jobs and return the (unstarted) scheduler."""
    sched = scheduler or BlockingScheduler()

    if edgar_tickers:
        sched.add_job(
            ingest_edgar_job,
            CronTrigger(hour=7, minute=30),
            args=[list(edgar_tickers), session_factory],
            id="edgar-daily",
            replace_existing=True,
        )

    if gdelt_queries:
        sched.add_job(
            ingest_gdelt_job,
            CronTrigger(minute="*/30"),  # news moves fast; poll every 30 min
            args=[list(gdelt_queries), session_factory],
            id="gdelt-news",
            replace_existing=True,
        )

    if market_tickers:
        sched.add_job(
            ingest_market_csv_job,
            CronTrigger(hour=18, minute=0),
            args=[list(market_tickers), session_factory, data_dir, market_dataset],
            id="market-csv-daily",
            replace_existing=True,
        )

    if fred_series:
        if not fred_api_key:
            log.warning("fred_series set but no FRED API key — skipping FRED job")
        else:
            sched.add_job(
                ingest_fred_job,
                CronTrigger(hour=8, minute=0),
                args=[list(fred_series), fred_api_key, session_factory],
                id="fred-daily",
                replace_existing=True,
            )

    return sched
