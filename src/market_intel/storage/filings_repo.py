"""Read/write SEC filings. Upserts idempotent on the accession_no PK."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from market_intel.storage.models import Filing


def upsert_filings(
    session: Session,
    filings: Sequence[dict],
    source: str = "SEC-EDGAR",
) -> int:
    """Upsert normalized filing dicts (see ingest.filings.parse_filings).

    Each dict must carry ``accession_no``, ``cik`` and ``form``. Returns rows written.
    """
    count = 0
    for f in filings:
        session.merge(
            Filing(
                accession_no=f["accession_no"],
                cik=f["cik"],
                ticker=f.get("ticker"),
                form=f["form"],
                filing_date=f.get("filing_date"),
                primary_doc=f.get("primary_doc"),
                description=f.get("description"),
                source=source,
            )
        )
        count += 1
    session.commit()
    return count


def get_filings(session: Session, ticker: str, limit: int = 50) -> list[Filing]:
    """Most-recent filings for a ticker first."""
    return list(
        session.scalars(
            select(Filing)
            .where(Filing.ticker == ticker.upper())
            .order_by(Filing.filing_date.desc().nulls_last())
            .limit(limit)
        ).all()
    )
