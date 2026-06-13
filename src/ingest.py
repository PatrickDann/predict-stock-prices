"""Thin CLI to ingest price CSVs into the database.

Requires a running Postgres (``docker compose up -d``) configured via .env.

Usage:
    python src/ingest.py AAPL
    python src/ingest.py MSFT AMZN --dataset tech
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_intel.ingest.market import ingest_from_csv  # noqa: E402
from market_intel.storage.db import init_db, make_engine, make_session_factory  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Ingest price CSVs into the database")
    p.add_argument("tickers", nargs="+", help="e.g. AAPL MSFT")
    p.add_argument("--dataset", default=None, help="file stem, e.g. 'tech'")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args(argv)

    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        for ticker in args.tickers:
            n = ingest_from_csv(session, ticker, data_dir=args.data_dir, dataset=args.dataset)
            print(f"Ingested {n} rows for {ticker.upper()}")


if __name__ == "__main__":
    main()
