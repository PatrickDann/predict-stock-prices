"""Pandera OHLCV validation gate."""

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from market_intel.ingest.validation import validate_ohlcv


def _frame(close, n=3):
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": [10.0] * n,
            "High": [11.0] * n,
            "Low": [9.0] * n,
            "Close": close,
            "Volume": [1000.0] * n,
        },
        index=idx,
    )


def test_valid_frame_passes():
    df = _frame([10.0, 11.0, 12.0])
    out = validate_ohlcv(df)
    assert len(out) == 3


def test_nonpositive_close_rejected():
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_ohlcv(_frame([10.0, 0.0, 12.0]))


def test_nan_close_rejected():
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_ohlcv(_frame([10.0, float("nan"), 12.0]))


def test_high_below_low_rejected():
    df = _frame([10.0, 11.0, 12.0])
    df.loc[df.index[0], "High"] = 1.0  # High < Low
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_ohlcv(df)


def test_non_datetime_index_rejected():
    df = _frame([10.0, 11.0, 12.0]).reset_index(drop=True)
    with pytest.raises(ValueError):
        validate_ohlcv(df)
