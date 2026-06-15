"""Fetch historical price CSVs from yfinance.

Replaces the original ``preprocess.py``, which had three ``__main__`` blocks
(all executed in sequence, and the third raised ``TypeError`` from a wrong-arity
call). Here each dataset is a declarative entry and ``main()`` fetches them all.

yfinance is unofficial / personal-use only and has no SLA — fine for prototyping;
Phase 1 adds a vetted source (Tiingo/Finnhub). See docs/VISION_AND_ROADMAP.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import yfinance as yf

DEFAULT_START = "2015-01-01"
DEFAULT_END = "2023-12-31"


@dataclass(frozen=True)
class Dataset:
    tickers: str  # space-separated, e.g. "AAPL MSFT"
    save_path: str
    start: str = DEFAULT_START
    end: str = DEFAULT_END


DATASETS: tuple[Dataset, ...] = (
    Dataset("AAPL", "data/aapl_stock_data.csv"),
    Dataset("AAPL MSFT GOOGL AMZN META NVDA", "data/tech_stock_data.csv"),
    Dataset("VOO VTI VUG VTV VYM", "data/index_fund_stock_data.csv"),
)


def fetch_stock_data(tickers: str, start: str, end: str, save_path: str) -> bool:
    """Download ``tickers`` between ``start`` and ``end`` and save to ``save_path``.

    Returns True if data was written, False on no/all-NaN data or any error.
    yfinance is unofficial and flaky, so failures are caught and reported rather
    than raised — one bad dataset shouldn't abort the rest.
    """
    try:
        data = yf.download(tickers, start=start, end=end)
    except Exception as exc:  # noqa: BLE001 — yfinance raises a variety of errors
        print(f"Failed to download {tickers!r}: {exc}")
        return False
    if data.empty or data.dropna(how="all").empty:
        print(f"No usable data for tickers {tickers!r}.")
        return False
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    data.to_csv(save_path)
    print(f"Data saved to {save_path}")
    return True


def main(datasets: tuple[Dataset, ...] = DATASETS) -> None:
    for ds in datasets:
        fetch_stock_data(ds.tickers, ds.start, ds.end, ds.save_path)


if __name__ == "__main__":
    main()
