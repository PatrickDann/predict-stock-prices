"""Thin CLI to run the ingestion scheduler.

Requires a running Postgres (``docker compose up -d``). FRED jobs need
``FRED_API_KEY`` in .env (free key: https://fredaccount.stlouisfed.org/apikey).

Usage:
    python src/worker.py --market AAPL MSFT --fred GDP CPIAUCSL
    python src/worker.py --dbnomics "IMF/WEO:latest/USA.NGDP_RPCH"   # keyless
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_intel.config import settings  # noqa: E402
from market_intel.scheduler import build_scheduler  # noqa: E402
from market_intel.storage.db import init_db, make_engine, make_session_factory  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run the ingestion scheduler")
    p.add_argument("--market", nargs="*", default=[], help="tickers to ingest from CSV")
    p.add_argument("--dataset", default=None, help="CSV dataset stem for --market")
    p.add_argument("--fred", nargs="*", default=[], help="FRED series ids, e.g. GDP CPIAUCSL")
    p.add_argument(
        "--dbnomics",
        nargs="*",
        default=[],
        help="DBnomics series ids, e.g. 'IMF/WEO:latest/USA.NGDP_RPCH' (no key)",
    )
    p.add_argument("--gdelt", nargs="*", default=[], help="GDELT queries, e.g. 'oil supply'")
    p.add_argument("--edgar", nargs="*", default=[], help="tickers for SEC EDGAR filings")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    sched = build_scheduler(
        session_factory,
        market_tickers=args.market,
        market_dataset=args.dataset,
        fred_series=args.fred,
        fred_api_key=settings.fred_api_key,
        dbnomics_series=args.dbnomics,
        gdelt_queries=args.gdelt,
        edgar_tickers=args.edgar,
        data_dir=str(settings.data_dir),
    )

    jobs = sched.get_jobs()
    if not jobs:
        print("No jobs configured. Pass --market and/or --fred (with FRED_API_KEY set).")
        return
    print("Scheduled jobs:")
    for job in jobs:
        print(f"  {job.id}: {job.trigger}")
    print("Starting scheduler (Ctrl-C to stop)...")
    sched.start()


if __name__ == "__main__":
    main()
