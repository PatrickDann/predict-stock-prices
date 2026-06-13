"""Pandera validation for OHLCV frames at the ingest boundary."""

from __future__ import annotations

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

# Prices must be positive (Close required); Volume non-negative; OHL nullable.
# coerce=True normalizes dtypes; strict=False tolerates extra columns.
OHLCV_SCHEMA = DataFrameSchema(
    {
        "Open": Column(float, Check.ge(0), nullable=True, required=False),
        "High": Column(float, Check.ge(0), nullable=True, required=False),
        "Low": Column(float, Check.ge(0), nullable=True, required=False),
        "Close": Column(float, Check.gt(0), nullable=False),
        "Volume": Column(float, Check.ge(0), nullable=True, required=False),
    },
    checks=Check(
        lambda df: (df["High"] >= df["Low"]) | df["High"].isna() | df["Low"].isna(),
        error="High must be >= Low",
    ),
    coerce=True,
    strict=False,
)


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate/coerce an OHLCV frame. Raises ``pandera.errors.SchemaError`` on
    violation. Also asserts a DatetimeIndex (what loaders/storage produce)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("OHLCV frame must have a DatetimeIndex")
    return OHLCV_SCHEMA.validate(df, lazy=False)
