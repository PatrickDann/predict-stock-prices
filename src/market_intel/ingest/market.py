"""Market-price ingestion: validate an OHLCV frame and upsert it.

``ingest_from_csv`` reads the committed yfinance CSVs (the Phase 0 source).
A live yfinance->DB path can wrap this with data.fetch in a later increment.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from market_intel.data.loaders import load_prices
from market_intel.ingest.validation import validate_ohlcv
from market_intel.storage.prices_repo import upsert_prices


def ingest_price_frame(
    session: Session,
    df: pd.DataFrame,
    symbol: str,
    source: str = "yfinance",
) -> int:
    """Validate then upsert a price frame. Returns rows written."""
    validated = validate_ohlcv(df)
    return upsert_prices(session, validated, symbol.upper(), source=source)


def ingest_from_csv(
    session: Session,
    ticker: str,
    data_dir: str | Path = "data",
    dataset: str | None = None,
    source: str = "yfinance-csv",
) -> int:
    """Load a ticker from a yfinance CSV and ingest it."""
    df = load_prices(ticker, data_dir=data_dir, dataset=dataset)
    return ingest_price_frame(session, df, ticker, source=source)
