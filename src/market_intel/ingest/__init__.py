"""Ingestion: fetch -> validate -> persist. Market prices first; macro/news next.

Validation gates (Pandera) sit at every ingest boundary so malformed data fails
loudly instead of silently corrupting downstream ML features.
"""

from market_intel.ingest.market import ingest_from_csv, ingest_price_frame

__all__ = ["ingest_from_csv", "ingest_price_frame"]
