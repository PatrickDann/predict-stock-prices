"""Read/write macroeconomic series. Upserts idempotent on (series_id, date)."""

from __future__ import annotations

import math

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from market_intel.storage.models import MacroSeries


def upsert_macro(
    session: Session,
    df: pd.DataFrame,
    series_id: str,
    source: str = "FRED",
) -> int:
    """Upsert a macro frame (DatetimeIndex, a ``value`` column) for ``series_id``.

    Rows with a NaN value are skipped. Returns the number of rows written.
    """
    count = 0
    for ts, row in df.iterrows():
        value = float(row["value"])
        if math.isnan(value):
            continue
        session.merge(
            MacroSeries(
                series_id=series_id,
                date=pd.Timestamp(ts).date(),
                value=value,
                source=source,
            )
        )
        count += 1
    session.commit()
    return count


def get_macro(session: Session, series_id: str) -> pd.DataFrame:
    """Return a series as a DataFrame indexed by date with a ``value`` column."""
    rows = session.scalars(
        select(MacroSeries).where(MacroSeries.series_id == series_id).order_by(MacroSeries.date)
    ).all()
    frame = pd.DataFrame(
        [{"Date": pd.Timestamp(r.date), "value": r.value} for r in rows],
        columns=["Date", "value"],
    )
    return frame.set_index("Date")
