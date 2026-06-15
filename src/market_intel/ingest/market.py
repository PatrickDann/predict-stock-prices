"""Market-price ingestion: validate an OHLCV frame and upsert it.

``ingest_from_csv`` reads the committed yfinance CSVs (the Phase 0 source);
``ingest_yfinance`` pulls live from yfinance. Both feed the same
validate -> upsert path.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from market_intel.data.loaders import DEFAULT_FIELDS, load_prices
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


def normalize_yf(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize a yfinance download into a tidy OHLCV frame.

    Robust to: flat columns or a MultiIndex in *either* level order
    ((field, ticker) or (ticker, field)); case-mismatched tickers; and
    tz-aware indexes (the trading-day date must stay stable).
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # Detect which level holds OHLCV field names; the other is the ticker.
        fields = {f.title() for f in DEFAULT_FIELDS}
        level0 = {str(x).strip().title() for x in df.columns.get_level_values(0)}
        level1 = {str(x).strip().title() for x in df.columns.get_level_values(1)}
        field_level = 0 if len(level0 & fields) >= len(level1 & fields) else 1
        ticker_level = 1 - field_level
        tick_values = list(dict.fromkeys(df.columns.get_level_values(ticker_level)))
        # Match the requested ticker case-insensitively; else take the first.
        match = next((v for v in tick_values if str(v).upper() == ticker.upper()), None)
        if match is None and tick_values:
            match = tick_values[0]
        df = df.xs(match, axis=1, level=ticker_level, drop_level=True)

    df = df.rename(columns={c: str(c).strip().title() for c in df.columns})
    available = [f for f in DEFAULT_FIELDS if f in df.columns]
    out = df[available].copy()

    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)  # keep local wall-clock date (the trading day)
    out.index = idx
    out.index.name = "Date"
    return out.dropna(subset=["Close"]) if "Close" in out.columns else out.dropna()


def ingest_yfinance(
    session: Session,
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    *,
    downloader: Callable[..., Any] | None = None,
    source: str = "yfinance",
) -> int:
    """Download a ticker live from yfinance and ingest it.

    ``downloader`` is injectable for testing; defaults to ``yfinance.download``.
    """
    if downloader is None:
        import yfinance as yf

        downloader = yf.download
    raw = downloader(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return 0
    df = normalize_yf(raw, ticker)
    return ingest_price_frame(session, df, ticker, source=source)
