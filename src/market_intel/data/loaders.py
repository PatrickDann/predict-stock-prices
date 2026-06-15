"""Format-agnostic loaders for yfinance price CSVs.

yfinance writes two shapes that the original single-format loader could not both
handle:

* single-ticker (``aapl_stock_data.csv``) — one ticker, 5 fields;
* multi-ticker (``tech_stock_data.csv``) — N tickers wide, 5 fields each.

Both share a 2-row column header ``(field, ticker)``. yfinance usually also
writes an index-name row (``Date,,,``) as the third line; we drop it *by
detecting non-parseable dates* rather than blindly skipping a fixed row, so a
file written without that row keeps all its real data.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

DEFAULT_FIELDS = ("Close", "High", "Low", "Open", "Volume")


def load_price_frame(path: str | Path) -> pd.DataFrame:
    """Parse a yfinance CSV into a DataFrame with a ``(field, ticker)`` column
    MultiIndex and a sorted ``DatetimeIndex``.

    Handles both single- and multi-ticker exports, strips header/value
    whitespace, drops any non-date index rows (incl. the spurious ``Date``
    index-name row *if present*), and coerces values to numeric.

    Raises ``FileNotFoundError`` if missing and ``ValueError`` if no rows parse.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")

    df = pd.read_csv(path, header=[0, 1], index_col=0)
    df.columns = pd.MultiIndex.from_tuples(
        [(str(field).strip(), str(ticker).strip()) for field, ticker in df.columns]
    )
    # Drop rows whose index isn't a real date — this removes the "Date,,," row
    # only when it exists, never a genuine first data row. We intentionally
    # coerce non-dates to NaT, so silence the mixed-format inference warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df.index.name = "Date"
    df = df.apply(pd.to_numeric, errors="coerce").sort_index()

    if df.empty:
        raise ValueError(f"No parseable date rows in {path} — unexpected file format")
    return df


def load_prices(
    ticker: str,
    data_dir: str | Path = "data",
    dataset: str | None = None,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
) -> pd.DataFrame:
    """Load a single ticker's OHLCV history as a tidy DataFrame.

    Parameters
    ----------
    ticker:
        e.g. ``"AAPL"``. Case-insensitive (both the filename and the in-file
        column match).
    data_dir:
        Directory containing the CSV files.
    dataset:
        Optional dataset name (e.g. ``"tech"`` -> ``tech_stock_data.csv``).
        Defaults to the per-ticker file ``<ticker>_stock_data.csv``.
    fields:
        Columns to return, in order. Defaults to Close/High/Low/Open/Volume.

    Returns
    -------
    DataFrame indexed by date with one column per field, rows with a missing
    value in any requested field dropped.
    """
    data_dir = Path(data_dir)
    stem = dataset if dataset is not None else ticker.lower()
    path = data_dir / f"{stem}_stock_data.csv"

    frame = load_price_frame(path)
    available = list(frame.columns.get_level_values(1).unique())
    canonical = {t.upper(): t for t in available}
    if ticker.upper() not in canonical:
        raise KeyError(f"Ticker {ticker!r} not in {path.name} (has: {available})")
    resolved = canonical[ticker.upper()]

    sub = frame.xs(resolved, axis=1, level=1)
    missing = [f for f in fields if f not in sub.columns]
    if missing:
        raise KeyError(f"Fields {missing} not in {path.name} for {resolved}")

    sub = sub[list(fields)]
    return sub.dropna(subset=list(fields))
