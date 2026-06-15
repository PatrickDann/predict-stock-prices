"""Thin CLI for a one-shot GDELT news pull (no API key needed).

Requires a running Postgres (``docker compose up -d``).

Usage:
    python src/ingest_news.py "oil supply disruption"
    python src/ingest_news.py "central bank" --timespan 3d --max-records 100
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_intel.ingest.news import ingest_gdelt  # noqa: E402
from market_intel.storage.db import init_db, make_engine, make_session_factory  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="One-shot GDELT news ingestion")
    p.add_argument("query", help="GDELT query string")
    p.add_argument("--timespan", default="1d", help="e.g. 1d, 3d, 1w")
    p.add_argument("--max-records", type=int, default=75)
    args = p.parse_args(argv)

    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        n = ingest_gdelt(session, args.query, timespan=args.timespan, max_records=args.max_records)
    print(f"Ingested {n} articles for query {args.query!r}")


if __name__ == "__main__":
    main()
