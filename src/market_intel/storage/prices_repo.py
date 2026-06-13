"""Read/write OHLCV bars. Upserts are idempotent on the (symbol, date) PK."""

from __future__ import annotations

import math

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from market_intel.storage.models import Price

_OHLC = ("Open", "High", "Low", "Close", "Volume")


def _f(value) -> float | None:
    if value is None:
        return None
    f = float(value)
    return None if math.isnan(f) else f


def _i(value) -> int | None:
    f = _f(value)
    return None if f is None else int(f)


def upsert_prices(
    session: Session,
    df: pd.DataFrame,
    symbol: str,
    source: str = "yfinance",
) -> int:
    """Upsert a price frame (DatetimeIndex, columns incl. Close) for ``symbol``.

    Uses ``session.merge`` so re-ingesting the same dates updates rather than
    duplicates — portable across Postgres and SQLite. Returns the row count.
    """
    count = 0
    for ts, row in df.iterrows():
        session.merge(
            Price(
                symbol=symbol,
                date=pd.Timestamp(ts).date(),
                open=_f(row.get("Open")),
                high=_f(row.get("High")),
                low=_f(row.get("Low")),
                close=_f(row["Close"]),
                volume=_i(row.get("Volume")),
                source=source,
            )
        )
        count += 1
    session.commit()
    return count


def get_prices(session: Session, symbol: str) -> pd.DataFrame:
    """Return a symbol's bars as a DataFrame (DatetimeIndex 'Date',
    columns Open/High/Low/Close/Volume) — symmetric with loaders.load_prices."""
    rows = session.scalars(select(Price).where(Price.symbol == symbol).order_by(Price.date)).all()
    frame = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp(r.date),
                "Open": r.open,
                "High": r.high,
                "Low": r.low,
                "Close": r.close,
                "Volume": r.volume,
            }
            for r in rows
        ],
        columns=["Date", *_OHLC],
    )
    return frame.set_index("Date")
