"""Thin CLI for a one-shot DBnomics macro-series pull (no API key needed).

Requires a running Postgres. A series id is the ``PROVIDER/DATASET/SERIES``
triple from db.nomics.world.

Usage:
    python src/ingest_dbnomics.py "IMF/WEO:latest/USA.NGDP_RPCH"
    python src/ingest_dbnomics.py "Eurostat/namq_10_gdp/Q.CLV10_MEUR.SCA.B1GQ.DE"
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_intel.ingest.dbnomics import ingest_dbnomics  # noqa: E402
from market_intel.storage.db import init_db, make_engine, make_session_factory  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="One-shot DBnomics macro ingestion")
    p.add_argument("series_ids", nargs="+", help="PROVIDER/DATASET/SERIES triples")
    args = p.parse_args(argv)

    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        for series_id in args.series_ids:
            n = ingest_dbnomics(session, series_id)
            print(f"Ingested {n} rows for {series_id}")


if __name__ == "__main__":
    main()
