"""Search ingested news — keyword (FTS) or semantic (vector).

Requires a running Postgres (``docker compose up -d``).

Usage:
    python src/search_news.py "oil supply"               # keyword
    python src/search_news.py "energy shock" --semantic  # vector similarity
    python src/search_news.py "rates" --semantic --embed # backfill embeddings first
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_intel.search import embed_pending, keyword_search, semantic_search  # noqa: E402
from market_intel.storage.db import init_db, make_engine, make_session_factory  # noqa: E402
from market_intel.storage.indexes import ensure_search_indexes  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Search ingested news")
    p.add_argument("query")
    p.add_argument("--semantic", action="store_true", help="vector similarity instead of keyword")
    p.add_argument("--embed", action="store_true", help="backfill missing embeddings first")
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args(argv)

    engine = make_engine()
    init_db(engine)
    ensure_search_indexes(engine)
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        if args.embed:
            n = embed_pending(session)
            print(f"(embedded {n} articles)")
        if args.semantic:
            results = semantic_search(session, args.query, limit=args.limit)
            for article, score in results:
                print(f"  [{score:.3f}] {article.seen_date}  {article.title}")
        else:
            for article in keyword_search(session, args.query, limit=args.limit):
                print(f"  {article.seen_date}  {article.title}")


if __name__ == "__main__":
    main()
