"""FRED macroeconomic-series ingestion.

Pipeline: fetch FRED observations -> parse to a frame -> validate -> upsert.
The HTTP fetch is isolated (and the ``get`` callable is injectable) so parsing
and ingestion are unit-testable without network or an API key.

FRED API: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
Free key: https://fredaccount.stlouisfed.org/apikey
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import requests
from sqlalchemy.orm import Session

from market_intel.ingest.validation import validate_macro
from market_intel.storage.macro_repo import upsert_macro

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred_observations(
    series_id: str,
    api_key: str,
    *,
    url: str = FRED_OBSERVATIONS_URL,
    get: Callable[..., Any] = requests.get,
    timeout: int = 30,
) -> dict:
    """Fetch raw observations JSON for a FRED series. Raises on HTTP error."""
    resp = get(
        url,
        params={"series_id": series_id, "api_key": api_key, "file_type": "json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def parse_fred_observations(payload: dict) -> pd.DataFrame:
    """Parse a FRED observations payload into a DataFrame.

    FRED encodes missing values as ``"."``; these become NaN and are dropped.
    Returns a DataFrame indexed by date with a single ``value`` column.
    """
    observations = payload.get("observations", [])
    frame = pd.DataFrame(
        [{"Date": o["date"], "value": o["value"]} for o in observations],
        columns=["Date", "value"],
    )
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")  # "." -> NaN
    frame = frame.dropna(subset=["Date", "value"]).set_index("Date").sort_index()
    return frame


def ingest_macro_frame(
    session: Session,
    df: pd.DataFrame,
    series_id: str,
    source: str = "FRED",
) -> int:
    """Validate then upsert a macro frame. Returns rows written."""
    validated = validate_macro(df)
    return upsert_macro(session, validated, series_id, source=source)


def ingest_fred(
    session: Session,
    series_id: str,
    api_key: str,
    *,
    get: Callable[..., Any] = requests.get,
) -> int:
    """Fetch a FRED series and ingest it. Needs a free FRED API key."""
    payload = fetch_fred_observations(series_id, api_key, get=get)
    df = parse_fred_observations(payload)
    return ingest_macro_frame(session, df, series_id, source="FRED")
