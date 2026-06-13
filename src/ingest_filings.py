"""Thin CLI for a one-shot SEC EDGAR filings pull (no API key needed).

Requires a running Postgres. Set SEC_USER_AGENT in .env to a descriptive
contact string (SEC blocks generic/empty agents).

Usage:
    python src/ingest_filings.py AAPL MSFT
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_intel.ingest.filings import ingest_edgar  # noqa: E402
from market_intel.storage.db import init_db, make_engine, make_session_factory  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="One-shot SEC EDGAR filings ingestion")
    p.add_argument("tickers", nargs="+", help="e.g. AAPL MSFT")
    args = p.parse_args(argv)

    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        for ticker in args.tickers:
            n = ingest_edgar(session, ticker)
            print(f"Ingested {n} filings for {ticker.upper()}")


if __name__ == "__main__":
    main()
