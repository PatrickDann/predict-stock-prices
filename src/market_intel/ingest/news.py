"""GDELT news ingestion — global, multilingual, free, no API key.

Pipeline: fetch GDELT DOC 2.0 ArtList -> parse/normalize (validation gate +
in-batch dedup) -> idempotent upsert. The HTTP fetch is isolated (injectable
``get``) so parsing/ingest are unit-testable without network.

GDELT DOC 2.0 API: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime
from typing import Any

import requests
from sqlalchemy.orm import Session

from market_intel.storage.news_repo import upsert_articles

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_gdelt_articles(
    query: str,
    *,
    max_records: int = 75,
    timespan: str = "1d",
    url: str = GDELT_DOC_URL,
    get: Callable[..., Any] = requests.get,
    timeout: int = 30,
) -> dict:
    """Fetch a GDELT DOC ArtList payload for ``query``. Raises on HTTP error."""
    resp = get(
        url,
        params={
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": max_records,
            "timespan": timespan,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _parse_seendate(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ")
    except ValueError:
        return None


def parse_gdelt_articles(payload: dict) -> list[dict]:
    """Normalize a GDELT payload into article dicts.

    Validation gate: skips records missing a url or title. Dedups within the
    batch by url_hash (GDELT often repeats the same story across outlets/queries).
    """
    by_hash: dict[str, dict] = {}
    for art in payload.get("articles") or []:
        url = art.get("url")
        title = art.get("title")
        if not url or not title:
            continue
        h = _url_hash(url)
        if h in by_hash:  # keep the first (typically more complete) record
            continue
        by_hash[h] = {
            "url_hash": h,
            "url": url,
            "title": title,
            "seen_date": _parse_seendate(art.get("seendate")),
            "domain": art.get("domain"),
            "language": art.get("language"),
            "source_country": art.get("sourcecountry"),
            "raw": art,
        }
    return list(by_hash.values())


def ingest_articles(session: Session, articles: list[dict], source: str = "GDELT") -> int:
    """Upsert already-parsed article dicts. Returns rows written."""
    return upsert_articles(session, articles, source=source)


def ingest_gdelt(
    session: Session,
    query: str,
    *,
    max_records: int = 75,
    timespan: str = "1d",
    get: Callable[..., Any] = requests.get,
) -> int:
    """Fetch a GDELT query and ingest its articles. No API key required."""
    payload = fetch_gdelt_articles(query, max_records=max_records, timespan=timespan, get=get)
    articles = parse_gdelt_articles(payload)
    return ingest_articles(session, articles, source="GDELT")
