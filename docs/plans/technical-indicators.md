# Plan — Technical indicators on the price chart (Phase 2)

_Roadmap item (§5, Phase 2): "Ticker-level sentiment on the news feed; **technical
indicators on the chart**." This change delivers the **technical-indicators** half._

> _Update: the ticker-level **sentiment** half shipped separately — see
> [news-sentiment.md](news-sentiment.md). It used a self-contained, dependency-free
> Loughran–McDonald-inspired lexicon baseline (FinBERT remains the Phase-3 drop-in),
> resolving the "not self-contained" concern noted below._
>
> The ticker-level **sentiment** half was initially deferred: it seemed to depend on
> FinBERT scoring (an explicit Phase-3 deliverable) or a keyed external sentiment
> API (Marketaux/Finnhub), neither of which is self-contained at this point. Technical
> indicators compute purely from price data already in Postgres, so they ship cleanly now.

## Goal

Compute the standard suite of technical indicators server-side from the OHLCV bars
already ingested, expose them over the FastAPI JSON API, and overlay them on the
dashboard's candlestick chart — matching the existing repo → compute → API → UI pattern.

## Indicators & conventions (verified against current practice)

All indicators are **causal** — every value uses only the current bar and trailing
bars (`rolling`/`ewm` never look forward), so no lookahead is introduced. Warm-up
periods (before a window is full) yield `NaN`, serialized as JSON `null` (consistent
with the existing `_num` NaN→None helper in `api/app.py`).

Computed from the **close** series (the standard input for these indicators):

| Indicator | Definition | Notes |
|---|---|---|
| **SMA(n)** | `close.rolling(n).mean()` | windows 20 and 50 |
| **EMA(n)** | `close.ewm(span=n, adjust=False).mean()` | `adjust=False` = the recursive EMA charting libs use; span 20 |
| **RSI(14)** | Wilder's smoothing: avg gain/loss via `ewm(alpha=1/period, adjust=False)`, `RSI = 100 − 100/(1+RS)` | avg_loss=0 → 100; avg_gain=0 → 0 |
| **MACD** | `EMA(12) − EMA(26)`; signal = `EMA(macd, 9)`; hist = `macd − signal` | all `adjust=False` |
| **Bollinger(20, 2σ)** | mid = SMA(20); upper/lower = mid ± 2·`rolling(20).std(ddof=0)` | `ddof=0` = classic (population) Bollinger std |

Defaults are fixed (standard windows) to keep the surface small and avoid untrusted
parameter parsing — the roadmap explicitly warns against over-engineering.

## Backend

- **`src/market_intel/indicators.py`** — flat module matching the existing
  `search.py` / `embeddings.py` style. Pure functions `sma`, `ema`, `rsi`, `macd`,
  `bollinger_bands` over a `pd.Series`, plus `compute_indicators(df) -> pd.DataFrame`
  that assembles the date-indexed indicator frame from a price frame
  (`get_prices`-shaped: DatetimeIndex + Close column).
- **`/api/indicators/{symbol}`** in `api/app.py` — fetch prices via the existing
  `get_prices`, compute over the **full** series, then `tail(limit)` so warm-up NaNs
  don't eat the displayed window. Returns a list of date-aligned records
  (`{date, sma_20, sma_50, ema_20, bb_upper, bb_mid, bb_lower, rsi_14, macd,
  macd_signal, macd_hist}`), each numeric value JSON-safe via the existing `_num`.
  `limit` query param mirrors `/api/prices` (default 500). Unknown symbol → `[]`.

## Frontend (`api/static/index.html`)

- A thin controls strip in the Price panel (styled like `.feed-controls`): toggles
  for **MA** (SMA20+SMA50), **BB** (Bollinger bands), **RSI** (bottom sub-scale).
- Overlay line series on the existing candlestick chart (price-unit scale): SMA20,
  SMA50, Bollinger upper/mid/lower.
- RSI rendered on a separate bottom price-scale within the same chart via
  `priceScaleId` + `scaleMargins` (toggleable; nudges the candles up when shown).
- A compact latest-value readout in the panel header (SMA20 / RSI / MACD-vs-signal)
  so MACD is surfaced without a second cramped sub-scale.
- `loadSymbol()` also triggers `loadIndicators(sym)`; toggles flip series `visible`.

## Tests (`tests/test_indicators.py` + extend `tests/test_api.py`)

- Known-value math checks: SMA/EMA on a hand-computable series; RSI bounds [0,100]
  and the all-gains→100 / all-losses→0 edge cases; MACD = EMA12−EMA26 and hist =
  macd−signal identities; Bollinger mid = SMA and band symmetry.
- Warm-up rows are `NaN` (e.g. first 19 SMA20 values).
- Causality guard: appending a future bar must not change earlier indicator values.
- API: `/api/indicators/AAPL` returns the expected keys, JSON-safe nulls in warm-up,
  honors `limit`, and `[]` for an unknown symbol.

## Out of scope

Configurable windows, ticker-level sentiment, intraday indicators, and persisting
indicators to the DB (they're cheap to recompute on read).

Sources: [ChartingLens — RSI/MACD/Bollinger guide (2026)](https://chartinglens.com/blog/rsi-macd-bollinger-bands-guide) ·
[pandas-ta-classic](https://pypi.org/project/pandas-ta-classic/) ·
[alpharithms — RSI in Python](https://www.alpharithms.com/relative-strength-index-rsi-in-python-470209/)
