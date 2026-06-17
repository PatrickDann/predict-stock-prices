"""SQLAlchemy ORM models.

No ``from __future__ import annotations`` here — SQLAlchemy 2.0 resolves the
``Mapped[...]`` annotations at class-definition time, and stringized annotations
can break that resolution.
"""

from datetime import date as date_type
from datetime import datetime as datetime_type

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, BigInteger, Date, DateTime, Float, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from market_intel.embeddings import EMBED_DIM

# JSONB on Postgres, generic JSON elsewhere (keeps SQLite-backed tests working).
JSON_OR_JSONB = JSON().with_variant(JSONB, "postgresql")
# pgvector Vector on Postgres; JSON list of floats on SQLite (for tests).
# none_as_null=True so an absent embedding is SQL NULL (matches IS NULL), not
# JSON 'null' — otherwise embedding.is_(None) filters would silently miss.
EMBEDDING_TYPE = Vector(EMBED_DIM).with_variant(JSON(none_as_null=True), "sqlite")


class Base(DeclarativeBase):
    pass


class Price(Base):
    """One daily OHLCV bar for a symbol. Composite PK makes upserts idempotent."""

    __tablename__ = "prices"
    __table_args__ = (Index("ix_prices_date", "date"),)

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    date: Mapped[date_type] = mapped_column(Date, primary_key=True)
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    source: Mapped[str] = mapped_column(String(40), default="yfinance", nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Price({self.symbol} {self.date} close={self.close})"


class MacroSeries(Base):
    """One observation of a macroeconomic series (e.g. FRED ``GDP``, ``CPIAUCSL``).

    Composite PK (series_id, date) makes re-ingestion idempotent.
    """

    __tablename__ = "macro_series"
    __table_args__ = (Index("ix_macro_series_date", "date"),)

    series_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    date: Mapped[date_type] = mapped_column(Date, primary_key=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(40), default="FRED", nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"MacroSeries({self.series_id} {self.date} value={self.value})"


class NewsArticle(Base):
    """A news article (initially from GDELT). PK is a hash of the URL so
    re-ingesting the same article updates rather than duplicates.

    ``raw`` keeps the full source record (JSONB on Postgres). A pgvector
    embedding column + tsvector FTS are added later when semantic search lands.
    """

    __tablename__ = "news_articles"
    __table_args__ = (Index("ix_news_seen_date", "seen_date"),)

    url_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    seen_date: Mapped[datetime_type | None] = mapped_column(DateTime, nullable=True)
    domain: Mapped[str | None] = mapped_column(String(255), nullable=True)
    language: Mapped[str | None] = mapped_column(String(40), nullable=True)
    source_country: Mapped[str | None] = mapped_column(String(80), nullable=True)
    raw: Mapped[dict | None] = mapped_column(JSON_OR_JSONB, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(EMBEDDING_TYPE, nullable=True)
    # Financial-sentiment polarity in [-1, 1]; NULL until scored (see sentiment.py).
    sentiment: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(40), default="GDELT", nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"NewsArticle({self.domain} {self.seen_date} {self.title[:40]!r})"


class Filing(Base):
    """An SEC EDGAR filing (10-K, 8-K, …). PK is the accession number, so
    re-ingesting a company's recent filings is idempotent."""

    __tablename__ = "filings"
    __table_args__ = (
        Index("ix_filings_filing_date", "filing_date"),
        Index("ix_filings_ticker", "ticker"),
    )

    accession_no: Mapped[str] = mapped_column(String(25), primary_key=True)
    cik: Mapped[str] = mapped_column(String(10), nullable=False)
    ticker: Mapped[str | None] = mapped_column(String(20), nullable=True)
    form: Mapped[str] = mapped_column(String(20), nullable=False)
    filing_date: Mapped[date_type | None] = mapped_column(Date, nullable=True)
    primary_doc: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(40), default="SEC-EDGAR", nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Filing({self.ticker or self.cik} {self.form} {self.filing_date})"
