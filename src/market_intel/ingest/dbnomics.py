"""DBnomics global macroeconomic-series ingestion.

DBnomics aggregates ~80 providers (World Bank, IMF, OECD, Eurostat, AMECO, …)
behind one free, keyless API — the global complement to the US-centric FRED
ingestor. It reuses the generic macro tail (``ingest_macro_frame`` -> validate
-> upsert); only the fetch+parse head is DBnomics-specific.

Pipeline: fetch one series -> parse to a frame -> validate -> upsert. The HTTP
fetch is isolated (and the ``get`` callable is injectable) so parsing and
ingestion are unit-testable without network.

A series is identified by the triple ``PROVIDER/DATASET/SERIES``, e.g.
``Eurostat/namq_10_gdp/Q.CLV10_MEUR.SCA.B1GQ.DE`` or ``IMF/WEO:latest/USA.NGDP_RPCH``.

DBnomics API: https://docs.db.nomics.world/web-api/
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import requests
from sqlalchemy.orm import Session

from market_intel.ingest.macro import ingest_macro_frame

DBNOMICS_SERIES_URL = "https://api.db.nomics.world/v22/series"


def fetch_dbnomics_series(
    series_id: str,
    *,
    url: str = DBNOMICS_SERIES_URL,
    get: Callable[..., Any] = requests.get,
    timeout: int = 30,
) -> dict:
    """Fetch a DBnomics series (with observations) as JSON. Raises on HTTP error.

    ``series_id`` is the full ``PROVIDER/DATASET/SERIES`` triple.
    """
    resp = get(
        url,
        params={"series_ids": series_id, "observations": 1},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def parse_dbnomics_series(payload: dict) -> pd.DataFrame:
    """Parse a DBnomics series payload into a DataFrame.

    Uses ``period_start_day`` (the ISO start date of each period) as the date so
    we never have to parse period labels like ``"1991-Q1"``. DBnomics encodes
    missing observations as the string ``"NA"``; these become NaN and are
    dropped. An empty/malformed payload yields an empty frame.

    Returns a DataFrame indexed by date with a single ``value`` column.
    """
    docs = payload.get("series", {}).get("docs", [])
    doc = docs[0] if docs else {}
    # period_start_day is a clean ISO date; fall back to the period label.
    dates = doc.get("period_start_day") or doc.get("period", [])
    values = doc.get("value", [])
    # The arrays are parallel; guard against a malformed payload misaligning them.
    n = min(len(dates), len(values))

    frame = pd.DataFrame({"Date": dates[:n], "value": values[:n]}, columns=["Date", "value"])
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")  # "NA" -> NaN
    frame = frame.dropna(subset=["Date", "value"]).set_index("Date").sort_index()
    return frame


def ingest_dbnomics(
    session: Session,
    series_id: str,
    *,
    get: Callable[..., Any] = requests.get,
) -> int:
    """Fetch a DBnomics series and ingest it. No API key required."""
    payload = fetch_dbnomics_series(series_id, get=get)
    df = parse_dbnomics_series(payload)
    return ingest_macro_frame(session, df, series_id, source="DBnomics")
