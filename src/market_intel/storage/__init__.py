"""Persistence layer: SQLAlchemy models, engine/session helpers, repositories.

Portable across Postgres (production) and SQLite (tests). Postgres-specific
optimizations (pgvector, native partitioning) are layered on separately and are
not required by the ORM models. Schema is created via ``init_db`` for now;
Alembic migrations come later in Phase 1.
"""

from market_intel.storage.db import init_db, make_engine, make_session_factory
from market_intel.storage.models import Base, MacroSeries, NewsArticle, Price

__all__ = [
    "Base",
    "MacroSeries",
    "NewsArticle",
    "Price",
    "init_db",
    "make_engine",
    "make_session_factory",
]
