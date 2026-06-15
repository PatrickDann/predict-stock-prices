"""News search: backfill embeddings, keyword (FTS) search, semantic search.

Dialect-aware so it works in both worlds:
- Postgres: full-text search via ``to_tsvector``/``websearch_to_tsquery`` and
  vector search via pgvector's ``<=>`` cosine distance (uses the GIN/HNSW
  indexes from ``storage.indexes.ensure_search_indexes``).
- SQLite (tests): ``LIKE`` keyword match and an in-Python cosine scan.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from market_intel.embeddings import Embedder, cosine_similarity, get_default_embedder
from market_intel.storage.models import NewsArticle


def _dialect(session: Session) -> str:
    return session.get_bind().dialect.name


def embed_pending(session: Session, embedder: Embedder | None = None, limit: int = 500) -> int:
    """Compute + store embeddings for articles that don't have one yet."""
    embedder = embedder or get_default_embedder()
    rows = list(
        session.scalars(
            select(NewsArticle).where(NewsArticle.embedding.is_(None)).limit(limit)
        ).all()
    )
    if not rows:
        return 0
    vectors = embedder.encode([r.title for r in rows])
    for row, vec in zip(rows, vectors, strict=True):
        row.embedding = vec
    session.commit()
    return len(rows)


def keyword_search(session: Session, query: str, limit: int = 20) -> list[NewsArticle]:
    """Full-text (Postgres) / LIKE (SQLite) search over article titles."""
    order = NewsArticle.seen_date.desc().nulls_last()
    if _dialect(session) == "postgresql":
        tsvector = func.to_tsvector("english", NewsArticle.title)
        tsquery = func.websearch_to_tsquery("english", query)
        stmt = select(NewsArticle).where(tsvector.op("@@")(tsquery)).order_by(order).limit(limit)
    else:
        stmt = (
            select(NewsArticle)
            .where(NewsArticle.title.ilike(f"%{query}%"))
            .order_by(order)
            .limit(limit)
        )
    return list(session.scalars(stmt).all())


def semantic_search(
    session: Session,
    query: str,
    embedder: Embedder | None = None,
    limit: int = 20,
) -> list[tuple[NewsArticle, float]]:
    """Nearest articles to ``query`` by cosine. Returns (article, similarity)
    pairs, most similar first. Only considers articles that have an embedding."""
    embedder = embedder or get_default_embedder()
    qvec = embedder.encode([query])[0]

    if _dialect(session) == "postgresql":
        distance = NewsArticle.embedding.cosine_distance(qvec)
        stmt = (
            select(NewsArticle, distance.label("distance"))
            .where(NewsArticle.embedding.is_not(None))
            .order_by(distance)
            .limit(limit)
        )
        return [(article, 1.0 - float(dist)) for article, dist in session.execute(stmt).all()]

    # SQLite fallback: in-Python cosine over stored JSON vectors.
    rows = session.scalars(select(NewsArticle).where(NewsArticle.embedding.is_not(None))).all()
    scored = [(r, cosine_similarity(qvec, r.embedding)) for r in rows]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:limit]
