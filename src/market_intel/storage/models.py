"""SQLAlchemy ORM models.

No ``from __future__ import annotations`` here — SQLAlchemy 2.0 resolves the
``Mapped[...]`` annotations at class-definition time, and stringized annotations
can break that resolution.
"""

from datetime import date as date_type

from sqlalchemy import BigInteger, Date, Float, Index, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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
