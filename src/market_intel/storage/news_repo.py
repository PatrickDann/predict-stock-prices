"""Read/write news articles. Upserts idempotent on the url_hash PK."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from market_intel.storage.models import NewsArticle


def upsert_articles(
    session: Session,
    articles: Sequence[dict],
    source: str = "GDELT",
) -> int:
    """Upsert normalized article dicts (see ingest.news.parse_gdelt_articles).

    Each dict must carry ``url_hash``, ``url`` and ``title``. Returns rows written.
    """
    count = 0
    for a in articles:
        session.merge(
            NewsArticle(
                url_hash=a["url_hash"],
                url=a["url"],
                title=a["title"],
                seen_date=a.get("seen_date"),
                domain=a.get("domain"),
                language=a.get("language"),
                source_country=a.get("source_country"),
                raw=a.get("raw"),
                source=source,
            )
        )
        count += 1
    session.commit()
    return count


def get_recent_articles(session: Session, limit: int = 50) -> list[NewsArticle]:
    """Most-recently-seen articles first (NULL seen_date last)."""
    return list(
        session.scalars(
            select(NewsArticle).order_by(NewsArticle.seen_date.desc().nulls_last()).limit(limit)
        ).all()
    )


def count_articles(session: Session) -> int:
    return session.query(NewsArticle).count()


def country_counts(session: Session) -> list[tuple[str, int]]:
    """Article counts grouped by ``source_country``, most active first.

    Rows with a NULL or empty ``source_country`` are excluded (nothing to place
    on a map). Returns ``[(country, count), ...]``.
    """
    n = func.count().label("n")
    return [
        (country, count)
        for country, count in session.execute(
            select(NewsArticle.source_country, n)
            .where(NewsArticle.source_country.isnot(None))
            .where(NewsArticle.source_country != "")
            .group_by(NewsArticle.source_country)
            .order_by(n.desc())
        ).all()
    ]
