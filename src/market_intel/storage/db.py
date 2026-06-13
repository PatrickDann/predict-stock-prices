"""Engine / session factory helpers."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from market_intel.config import settings
from market_intel.storage.models import Base


def make_engine(url: str | None = None, echo: bool = False, **kwargs) -> Engine:
    """Create an Engine. Defaults to the configured Postgres URL; pass a
    ``sqlite:///...`` URL for tests."""
    return create_engine(url or settings.database_url, echo=echo, future=True, **kwargs)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(engine: Engine) -> None:
    """Create all tables that don't yet exist (no migrations — Alembic later)."""
    Base.metadata.create_all(engine)
