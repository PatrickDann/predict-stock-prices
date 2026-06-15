"""Math + causality checks for the technical-indicators module."""

import numpy as np
import pandas as pd

from market_intel.indicators import (
    bollinger_bands,
    compute_indicators,
    ema,
    macd,
    rsi,
    sma,
)


def _frame(closes) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"Close": [float(c) for c in closes]}, index=idx)


def test_sma_known_values_and_warmup():
    s = pd.Series([1.0, 2, 3, 4, 5])
    out = sma(s, 2)
    assert np.isnan(out.iloc[0])  # warm-up
    assert list(out.iloc[1:]) == [1.5, 2.5, 3.5, 4.5]


def test_ema_starts_at_first_value_and_stays_bounded():
    s = pd.Series([10.0, 20, 30, 40])
    out = ema(s, span=3)
    assert out.iloc[0] == 10.0  # adjust=False seeds on the first value
    assert out.is_monotonic_increasing
    assert out.max() <= s.max() and out.min() >= s.min()


def test_rsi_all_gains_is_100():
    out = rsi(pd.Series(np.arange(1.0, 30.0)))
    assert np.isnan(out.iloc[0])
    assert np.allclose(out.iloc[1:], 100.0)


def test_rsi_all_losses_is_0():
    out = rsi(pd.Series(np.arange(30.0, 1.0, -1.0)))
    assert np.allclose(out.iloc[1:], 0.0)


def test_rsi_stays_in_bounds():
    rng = np.random.default_rng(0)
    walk = 100 + np.cumsum(rng.normal(0, 1, 200))
    out = rsi(pd.Series(walk)).dropna()
    assert (out >= 0).all() and (out <= 100).all()


def test_macd_identities():
    rng = np.random.default_rng(1)
    s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 120)))
    out = macd(s)
    assert np.allclose(out["macd"], ema(s, 12) - ema(s, 26))
    assert np.allclose(out["macd_hist"], out["macd"] - out["macd_signal"])


def test_bollinger_mid_is_sma_and_bands_symmetric():
    rng = np.random.default_rng(2)
    s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 60)))
    bb = bollinger_bands(s, window=20, num_std=2.0)
    assert np.allclose(bb["bb_mid"].dropna(), sma(s, 20).dropna())
    width_up = (bb["bb_upper"] - bb["bb_mid"]).dropna()
    width_dn = (bb["bb_mid"] - bb["bb_lower"]).dropna()
    assert np.allclose(width_up, width_dn)


def test_compute_indicators_columns_and_warmup():
    df = _frame(100 + np.cumsum(np.ones(80)))
    out = compute_indicators(df)
    assert set(out.columns) == {
        "sma_20",
        "sma_50",
        "ema_20",
        "bb_upper",
        "bb_mid",
        "bb_lower",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
    }
    assert out["sma_20"].iloc[:19].isna().all()  # warm-up
    assert out["sma_20"].iloc[19:].notna().all()
    assert out.index.equals(df.index)


def test_compute_indicators_is_causal():
    base = _frame(100 + np.cumsum(np.sin(np.arange(60))))
    extended = _frame(list(base["Close"]) + [999.0])  # a wild future bar
    a = compute_indicators(base)
    b = compute_indicators(extended).iloc[: len(base)]
    pd.testing.assert_frame_equal(a, b)


def test_compute_indicators_empty_frame():
    empty = pd.DataFrame({"Close": []}, index=pd.DatetimeIndex([]))
    assert compute_indicators(empty).empty


def test_rsi_period_changes_smoothing():
    # Both periods only warm up row 0, so they align — but smooth differently.
    s = pd.Series(100 + np.cumsum(np.random.default_rng(3).normal(0, 1, 100)))
    assert not np.allclose(rsi(s, 7).dropna(), rsi(s, 21).dropna())
