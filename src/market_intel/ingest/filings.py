"""SEC EDGAR filings ingestion — free, no API key (descriptive User-Agent required).

Pipeline: resolve ticker -> CIK (company_tickers.json) -> fetch submissions ->
parse recent filings -> idempotent upsert. HTTP is isolated (injectable ``get``)
so resolution/parsing/ingest are unit-testable without network.

SEC requires a User-Agent with contact info: https://www.sec.gov/os/webmaster-faq#developers
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

import requests
from sqlalchemy.orm import Session

from market_intel.config import settings
from market_intel.storage.filings_repo import upsert_filings

log = logging.getLogger(__name__)

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"


def _get_json(url: str, *, get: Callable[..., Any], user_agent: str, timeout: int = 30) -> Any:
    resp = get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def resolve_cik(
    ticker: str,
    *,
    get: Callable[..., Any] = requests.get,
    user_agent: str | None = None,
) -> str | None:
    """Map a ticker to its 10-digit zero-padded CIK, or None if not found."""
    payload = _get_json(
        COMPANY_TICKERS_URL, get=get, user_agent=user_agent or settings.sec_user_agent
    )
    target = ticker.upper()
    for row in payload.values():
        cik_str = row.get("cik_str")
        if str(row.get("ticker", "")).upper() == target and cik_str is not None:
            return f"{int(cik_str):010d}"
    return None


def _parse_date(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_filings(payload: dict, ticker: str | None = None) -> list[dict]:
    """Parse an EDGAR submissions payload's ``filings.recent`` parallel arrays.

    Validation gate: skips entries missing an accession number or form.
    """
    cik = f"{int(payload['cik']):010d}" if str(payload.get("cik", "")).strip() else ""
    recent = (payload.get("filings") or {}).get("recent") or {}
    accessions = recent.get("accessionNumber") or []
    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    docs = recent.get("primaryDocument") or []
    descs = recent.get("primaryDocDescription") or []

    if len({len(accessions), len(forms), len(dates), len(docs), len(descs)}) > 1:
        log.warning(
            "EDGAR parallel arrays differ in length for cik %s (acc=%d form=%d date=%d) — "
            "positional pairing may be skewed",
            cik or "?",
            len(accessions),
            len(forms),
            len(dates),
        )

    out: list[dict] = []
    for i, accession in enumerate(accessions):
        accession = (accession or "").strip()
        form = forms[i] if i < len(forms) else None
        if not accession or not form:
            continue
        out.append(
            {
                "accession_no": accession,
                "cik": cik,
                "ticker": ticker.upper() if ticker else None,
                "form": form,
                "filing_date": _parse_date(dates[i] if i < len(dates) else None),
                "primary_doc": docs[i] if i < len(docs) else None,
                "description": descs[i] if i < len(descs) else None,
            }
        )
    return out


def ingest_filings(session: Session, filings: list[dict], source: str = "SEC-EDGAR") -> int:
    """Upsert already-parsed filing dicts. Returns rows written."""
    return upsert_filings(session, filings, source=source)


def ingest_edgar(
    session: Session,
    ticker: str,
    *,
    get: Callable[..., Any] = requests.get,
    user_agent: str | None = None,
) -> int:
    """Resolve a ticker's CIK, fetch its recent filings, and ingest them."""
    ua = user_agent or settings.sec_user_agent
    cik = resolve_cik(ticker, get=get, user_agent=ua)
    if cik is None:
        raise KeyError(f"No CIK found for ticker {ticker!r}")
    payload = _get_json(SUBMISSIONS_URL.format(cik=cik), get=get, user_agent=ua)
    filings = parse_filings(payload, ticker=ticker)
    return ingest_filings(session, filings)
