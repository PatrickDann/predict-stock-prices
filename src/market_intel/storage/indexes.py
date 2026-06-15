"""Postgres-only search indexes (GIN full-text + HNSW vector).

Created after the tables exist (``init_db``), not at container init. No-op on
SQLite. Idempotent (IF NOT EXISTS), safe to call on every startup.
"""

from __future__ import annotations

from sqlalchemy import Engine, text

_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS ix_news_title_fts "
    "ON news_articles USING gin (to_tsvector('english', title))",
    "CREATE INDEX IF NOT EXISTS ix_news_embedding_hnsw "
    "ON news_articles USING hnsw (embedding vector_cosine_ops)",
)


def ensure_search_indexes(engine: Engine) -> None:
    """Create the news FTS + vector indexes on Postgres; no-op elsewhere."""
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        for stmt in _STATEMENTS:
            conn.execute(text(stmt))
