"""Ingestion: fetch -> validate -> persist. Market prices first; macro/news next.

Validation gates (Pandera) sit at every ingest boundary so malformed data fails
loudly instead of silently corrupting downstream ML features.
"""

from market_intel.ingest.macro import (
    ingest_fred,
    ingest_macro_frame,
    parse_fred_observations,
)
from market_intel.ingest.market import ingest_from_csv, ingest_price_frame
from market_intel.ingest.news import (
    ingest_articles,
    ingest_gdelt,
    parse_gdelt_articles,
)

__all__ = [
    "ingest_from_csv",
    "ingest_price_frame",
    "ingest_fred",
    "ingest_macro_frame",
    "parse_fred_observations",
    "ingest_gdelt",
    "ingest_articles",
    "parse_gdelt_articles",
]
