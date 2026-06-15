"""Technical indicators over an OHLCV price frame.

All indicators are **causal**: every value depends only on the current bar and
trailing bars (``rolling``/``ewm`` never look forward), so none introduce
lookahead. Warm-up rows — before a window is full — are ``NaN`` and surface as
JSON ``null`` once serialized.

Conventions follow standard charting libraries:
- EMA/MACD use ``ewm(..., adjust=False)`` (the recursive EMA traders expect).
- RSI uses Wilder's smoothing (``ewm(alpha=1/period, adjust=False)``).
- Bollinger Bands use the population std (``ddof=0``), the classic definition.
"""

from __future__ import annotations

import pandas as pd

# Standard window defaults (kept fixed — no untrusted parameter parsing).
SMA_WINDOWS = (20, 50)
EMA_SPAN = 20
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
BB_NUM_STD = 2.0


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over ``window`` bars."""
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (recursive form, ``adjust=False``)."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Returns values in [0, 100]. A flat-or-falling window (no average gain)
    yields 0; a flat-or-rising window (no average loss) yields 100.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing == EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # avg_loss == 0 -> rs is +inf -> out is already 100; force avg_gain == 0 -> 0.
    # Row 0 stays NaN: the first .diff() is NaN, so avg_gain[0] is NaN (not 0) and
    # the .where below keeps it — the first bar is warm-up.
    return out.where(avg_gain != 0, 0.0)


def macd(
    series: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram as a 3-column frame."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_line - signal_line,
        }
    )


def bollinger_bands(
    series: pd.Series,
    window: int = BB_WINDOW,
    num_std: float = BB_NUM_STD,
) -> pd.DataFrame:
    """Bollinger Bands (upper/mid/lower) using the population std (``ddof=0``)."""
    mid = sma(series, window)
    std = series.rolling(window).std(ddof=0)
    return pd.DataFrame(
        {
            "bb_upper": mid + num_std * std,
            "bb_mid": mid,
            "bb_lower": mid - num_std * std,
        }
    )


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the standard indicator suite from a price frame.

    ``df`` is ``get_prices``/``load_prices``-shaped: a DatetimeIndex with a
    ``Close`` column. Returns a frame on the same index with one column per
    indicator series. An empty input yields an empty frame.
    """
    close = df["Close"].astype("float64")
    out = pd.DataFrame(index=df.index)
    for window in SMA_WINDOWS:
        out[f"sma_{window}"] = sma(close, window)
    out[f"ema_{EMA_SPAN}"] = ema(close, EMA_SPAN)
    out = out.join(bollinger_bands(close))
    out[f"rsi_{RSI_PERIOD}"] = rsi(close)
    out = out.join(macd(close))
    return out
