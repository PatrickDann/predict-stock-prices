# Global Market Intelligence Terminal — Vision & Roadmap

> From a single-file LSTM stock-price demo to a personal, Bloomberg-style global
> intelligence dashboard that ingests news, economics, and markets worldwide —
> and feeds a clean, leakage-free ML forecasting layer.

_Last updated: 2026-06-13. Research current as of mid-2026 — verify vendor pricing/limits before committing._

---

## 1. Where we are today (codebase review)

The repo is an early-stage hobby project: **2 Python scripts + 3 CSVs, 8 tracked files.**

```
predict-stock-prices/
├── requirements.txt          # ⚠️ missing the actual ML deps
├── README.md                 # ⚠️ truncated mid-sentence
├── .gitignore
├── data/                     # 3 yfinance CSV dumps (tracked in git)
│   ├── aapl_stock_data.csv         (single-ticker, wide)
│   ├── tech_stock_data.csv         (multi-ticker, different shape)
│   └── index_fund_stock_data.csv
└── src/
    ├── preprocess.py         # yfinance download → CSV
    └── multi_feature_model.py# load CSV → LSTM → predict → matplotlib
```

### What works
- A clean, readable end-to-end LSTM example: fetch → normalize → window → train → evaluate (RMSE) → plot.
- Sensible model shape: 2× LSTM(50) + Dropout + Dense, multivariate input (Close/High/Low/Open/Volume), 60-step lookback.
- Multi-ticker data already collected (tech + index funds).

### Concrete issues found (fix these in Phase 0)

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| 1 | **High** | `requirements.txt` omits `tensorflow`, `scikit-learn`, `matplotlib`. A fresh `pip install -r requirements.txt` cannot run the model. | `requirements.txt` |
| 2 | **High — ML correctness** | **Data leakage.** `MinMaxScaler` is fit on the *entire* dataset before the train/val/test split (`scaled_data = scaler.fit_transform(...)`), and `scaler_close` is also fit on the full series. Windows are also built over the full array *before* splitting, so windows near boundaries peek across them. This inflates reported RMSE — the model looks better than it is. | `multi_feature_model.py:24-44, 68-69` |
| 3 | **Medium** | `preprocess.py` has **three `if __name__ == "__main__"` blocks**; Python runs all three in sequence. The third (index funds) call passes only 3 args to a 4-arg function → `TypeError` at the end of every run. | `preprocess.py:15-34` |
| 4 | **Medium** | The CSV loader is hardcoded to the single-ticker format (`skiprows=2`, fixed column rename). It cannot parse `tech_stock_data.csv` / `index_fund_stock_data.csv` (multi-ticker wide format), so `multi_feature_model.py TECH` breaks. | `multi_feature_model.py:18-19` |
| 5 | **Medium** | No validation split discipline beyond fixed slices; no walk-forward / `TimeSeriesSplit`; hyperparameters not isolated to train. | `multi_feature_model.py:39-44` |
| 6 | **Low** | Everything in one script (data + model + viz). No package layout, no config, no secrets handling, no saved/versioned models, no tests. | `src/` |
| 7 | **Low** | Growing time-series stored as CSVs tracked in git — fine now, won't scale. | `data/` |
| 8 | **Low** | README truncated; no run instructions, no architecture doc. | `README.md` |

> Repo hygiene note: `.venv` (1.6 GB) and `sklearn-env` (172 MB) live in the working tree and inflate it to ~1.9 GB. Neither is tracked (good), but only `.venv/`/`venv/` are in `.gitignore` — add `sklearn-env/` and standardize on **one** environment.

---

## 2. Where we want to be (the end goal)

A **personal Bloomberg-style terminal** that:

- **Ingests globally and continuously** — news (incl. obscure, non-English, edge-of-the-world events), macro/economic releases, equities/FX/crypto/commodities.
- **Stores everything cleanly** in a queryable, ML-ready data layer.
- **Displays a dense, dark, multi-panel UI** — live news feed, candlestick charts, economic indicators, a world map of events, and model forecasts.
- **Runs a clean, leakage-free ML layer** (LSTM today; TFT/N-HiTS tomorrow) that *fuses* news sentiment + macro + price.
- **Surfaces signals** — links a faint headline in a remote market to the assets it might move.

### The gap, at a glance

| Capability | Today | End goal |
|---|---|---|
| Data sources | 1 (yfinance CSV) | News + macro + multi-asset markets + social |
| Ingestion | Manual script run | Scheduled, validated, continuous pipeline |
| Storage | 3 CSVs in git | Postgres/TimescaleDB + Parquet + search/vector |
| Coverage | US equities | Global, multilingual, real-time |
| UI | matplotlib `plt.show()` | Live multi-panel web terminal |
| ML rigor | Leaky single LSTM | Leakage-free, walk-forward, tracked, multi-model |
| Predictions | Ad-hoc plot | Scheduled forecasts stored + displayed |
| News→market link | None | Sentiment/entity fusion + event signals |

---

## 3. Recommended tech stack (researched, 2026)

**Guiding principle:** stay *boring and Postgres-centric*. Almost every "scalable" tool (Kafka, Airflow, Feast, a vector-DB cluster) is overkill at Phase 1. The lean stack runs on a laptop or a ~$5–10/mo VPS and scales **additively** — you swap one layer at a time, never rewrite.

### Phase-1 lean stack (laptop / one cheap VPS)

| Layer | Pick | Why |
|---|---|---|
| **Market history (ML-grade)** | **Tiingo** (~$30/mo flat, adjusted EOD back to ~1962) + **yfinance** for prototyping only | Clean, adjusted, survivorship-aware series — the one paid line item worth it for ML |
| **Real-time + streaming quotes** | **Finnhub free** (60 req/min, real-time US, WebSocket 50 symbols) | Best free real-time + websocket |
| **Macro / economics** | **FRED API** (free) + **DBnomics** (free, aggregates World Bank/IMF/OECD/Eurostat) | Covers ~95% of macro needs for $0 |
| **Global news / events** | **GDELT 2.0 DOC API + GKG** (free, 100+ languages, 15-min, redistribution allowed) + self-hosted **RSS via FreshRSS** | GDELT = unbeatable global breadth + the edge-of-world angle; RSS = near-real-time long tail |
| **Ticker-level sentiment** | **Marketaux** or **Finnhub** (free tiers) | Native per-entity sentiment, no NLP needed |
| **Social signal (optional)** | **Bluesky Jetstream** (free websocket) + **Reddit** (free) | On-the-ground early signals; X/Twitter API is cost-prohibitive in 2026 |
| **Filings / fundamentals** | **SEC EDGAR API** (free, survivorship-bias-free) | Straight from source |
| **Database (time-series + docs + vector)** | **Self-hosted Postgres + pgvector** (`pgvector/pgvector:pg17` in Docker) on hardware you own — **`pgvector` HNSW + `tsvector` FTS + JSONB**, time-series via **native partitioning** (BRIN now → `pg_partman`+`pg_cron` when large). _See §3.1._ | Genuinely **$0**, no row/storage caps, real concurrency (poller writes while dashboard reads), full extension freedom |
| **ML/analytics layer** | **DuckDB over Parquet** snapshots (can `ATTACH` the Postgres DB directly) | In-process, fast feature engineering + training inputs; **archiving history to Parquet keeps the hot DB small** |
| **Live push** | **FastAPI WebSockets/SSE** (Redis pub/sub for fan-out later) | Self-hosted, no managed realtime service needed |
| **Backups + remote access** | Nightly `pg_dump -Fc` → **Cloudflare R2 / Backblaze B2** (free tiers); DB stays **LAN-only or on Tailscale** | Off-box disaster recovery; never expose port 5432 publicly |
| **Orchestration** | **APScheduler** in the Python worker (heavy ingestion/ML); **pg_cron** for light in-DB jobs | Zero extra infra; poll on schedules |
| **Data validation** | **Pandera** at ingest + pre-train | Lightweight DataFrame contracts; quarantine bad rows |
| **Dataset versioning** | **DVC + dated Parquet snapshots** | Reproducible training sets without a feature store |
| **Backend API** | **FastAPI** + WebSockets/SSE | Async, typed, live updates; *doubles as model-serving + dashboard backend* |
| **Cache / async work** | **Redis** + **ARQ** | Async-native worker matching FastAPI |
| **ML modeling** | **Darts** (LSTM baseline, swap to N-HiTS/TFT later) | One API for LSTM → N-HiTS → TFT |
| **Experiment tracking** | **MLflow** (local) + model registry | Free, self-hosted |
| **Sentiment NLP** | **FinBERT** (self-hosted); LLM only on top-N flagged stories | Cheap bulk scoring; LLM for nuance |
| **UI** | **OpenBB Workspace (free) + FastAPI custom backend** *or* **Streamlit** | Fastest path to a dense dark terminal |
| **Charts** | **TradingView Lightweight Charts** (candles) + **ECharts** (everything else) | Best-in-class free financial charting |
| **Map** | **Leaflet** (Phase 1) → **deck.gl/Kepler.gl** (Phase 2) | World-event visualization |
| **Deploy** | **All self-hosted via `docker-compose`** (Postgres+pgvector, Redis, Python worker) on an **Oracle Cloud Always Free** VM (4 ARM cores / 24 GB RAM, PAYG = $0); `restart: unless-stopped`. Owned hardware or Xata free are interchangeable alternatives (see §3.1). | Truly free, always-on; identical dev/prod |

**Phase-1 monthly cost: $0** (self-hosted DB + all-free data sources: yfinance, Finnhub free, FRED, GDELT, SEC EDGAR). The only optional paid line item is **Tiingo (~$30/mo)** for cleaner ML-grade history — defer it; start on yfinance and add it only if data quality bites. Real cost is electricity + your time.

### Scaling stack (add only when a real bottleneck appears)
- Database: self-hosted Postgres → add **`pg_partman` partitioning** as tables grow. Since you self-host, **TimescaleDB is now an option** (use a combined `timescaledb`+`pgvector` image) — but at this scale native partitioning + DuckDB/Parquet is simpler and sufficient (see §3.1). If you ever want a managed always-on DB instead, **Xata free** (15 GB, no pause) or a ~$5/mo VPS/managed Postgres.
- Orchestration: APScheduler → **Prefect** (free Cloud Hobby / self-host) when jobs gain dependencies/backfills.
- Streaming: add **NATS JetStream** or **Redpanda** *only* if you move from polling to high-rate event ingestion.
- Analytics store: DuckDB/Parquet → **ClickHouse** (billions of rows) or **QuestDB** (tick data).
- Search: pgvector → **Qdrant** (10M+ vectors) and/or **Meilisearch/Typesense** (keyword relevance).
- Data sources: add **Polygon/"Massive"** ($29–$199, intraday + flat files) and **EODHD** ($100, global), **NewsData.io** ($200, commercial multilingual).
- Models: **N-HiTS** (cheap+strong) / **TFT** (covariates + quantile forecasts) via Darts or **NeuralForecast (Nixtla)**.
- Serving: FastAPI batch → **BentoML** for real-time inference with batching.
- UI: Streamlit/OpenBB → **Next.js + Lightweight Charts/ECharts/deck.gl** against the same FastAPI backend.

### 3.1 Database decision: self-hosted Postgres + pgvector (free), **not** Supabase, **not** TimescaleDB

**Decision:** self-host **plain PostgreSQL + pgvector** in Docker on hardware you own. This is the only genuinely-free option that handles a **24/7 ingestion** workload — verified against current (June 2026) docs.

**Why not Supabase / a free managed tier?** Free managed tiers are built for *idle* apps that sleep; a 24/7 poller is the opposite:
- **Supabase Free** — 500 MB cap is the blocker (constant writes prevent the 7-day pause, but the storage fills in days). Pro is ~$25/mo. Also, Supabase dropped `timescaledb` on its PG17 default.
- **Neon Free** — pgvector ✓, but **100 compute-hours/mo ≈ 400 h** of a ~730 h month; an always-awake poller exhausts it ~day 17, then compute suspends.
- **Xata Free** (15 GB, no pause/cold-start) is the best *managed* free fallback if you ever stop self-hosting.

**Why this is great at our scale:** one Docker container (`pgvector/pgvector:pg17`), true MVCC (poller writes while dashboard reads — no single-writer bottleneck), real HNSW/IVFFlat vector indexes, full `tsvector` FTS + JSONB, **no row/storage caps** (bounded only by disk), and complete extension freedom. Cost: $0 + electricity.

**Time-series strategy (no TimescaleDB needed):**
- **Start:** one plain table per series + **BRIN index** on the timestamp (tiny, zero maintenance).
- **Grow:** `pg_partman` monthly range partitions + `pg_cron` for auto-create/retention.
- **ML/aggregates + keeping the DB small:** snapshot older history to **Parquet + DuckDB** (free, on disk); keep only a hot rolling window in Postgres. DuckDB covers compression and fast time-bucketed aggregation; continuous aggregates → a `pg_cron`-refreshed materialized view.
- Since you self-host, **TimescaleDB is available** (combined `timescaledb`+`pgvector` image) if you later want hypertables — but it's unnecessary at personal scale.

**Host of record: Oracle Cloud Always Free (PAYG, $0).** Rather than depend on a home box, run the whole stack on an **Oracle Cloud Always Free** VM — **4 ARM (Ampere A1) cores / 24 GB RAM / 200 GB storage / 10 TB egress**, free indefinitely. Co-locate Postgres + Redis + the Python worker on the one VM so the DB never leaves it (the `127.0.0.1:5432` binding stays as-is); reach only the dashboard remotely over Tailscale. Provisioning checklist:
- **Convert the account to Pay-As-You-Go (PAYG)** — disables idle-instance reclamation; you still pay **$0** within Always-Free limits. (Without PAYG, an idle VM can be reclaimed; PAYG is the reliable fix.)
- Pick a **home region with A1 capacity** (it's fixed after signup); retry the create / use an `oci-arm-host-capacity` script if "Out of Capacity".
- Install Docker + Tailscale; `git clone`; `docker compose up -d` — **same files, zero code change**.

**Operational checklist (the honest ops burden):**
- `restart: unless-stopped` in compose (already set); the VM is always-on.
- **Backups (non-negotiable — no managed backups, real account-termination tail risk):** nightly `pg_dump -Fc` → **Cloudflare R2 / Backblaze B2** (free tiers); keep a rolling window. **Treat the VM as disposable** — a reclaim or ban should be a restore, not a loss.
- **Exposure:** keep Postgres **LAN/tailnet-only**; never publish port 5432. Use a Cloudflare Tunnel only for the dashboard's web UI.
- A **Python worker (FastAPI + APScheduler)** does heavy ingestion (GDELT, FinBERT) and ML; live updates go out over **FastAPI WebSockets/SSE** (add Redis pub/sub for fan-out later).

**Alternatives:** **a machine you own** (Mac mini / old PC / Pi 5 — same stack, no capacity lottery, but you supply uptime/power), or **Xata free** managed (15 GB, no pause). The architecture is identical (it's all just Postgres), so switching is trivial. For local dev, `docker compose up` on your laptop mirrors the VM exactly.

Sources: [timescaledb extension (deprecation notice)](https://supabase.com/docs/guides/database/extensions/timescaledb) · [PG17 release notes / removal](https://supabase.com/changelog/35851-forthcoming-postgres-17-release-notes) · [self-hosted PG15→17 breaking change](https://supabase.com/changelog/46080-self-hosted-supabase-upgrading-from-pg-15-to-17-breaking-change) · [pg_partman](https://supabase.com/docs/guides/database/extensions/pg_partman) · [Realtime benchmarks](https://supabase.com/docs/guides/realtime/benchmarks) · [pricing](https://supabase.com/pricing).

### Notable shortcut
**OpenBB** (open-source) solves two problems at once: its **Platform** is a free Python API over ~100 data providers (replaces ad-hoc yfinance), and **Workspace Community Edition** is a free, dense, dark, draggable terminal UI where your own FastAPI endpoints (price, news, macro, forecasts) appear as widgets via a `widgets.json` spec — **no frontend code required.**

---

## 4. Target architecture

```
                        ┌─────────────────── INGESTION (APScheduler jobs) ───────────────────┐
  External sources  →   │  market_poller   macro_poller   news_poller   social_consumer      │
  (Tiingo/Finnhub,      │       │              │              │              │                │
   FRED/DBnomics,       │       └──────┬───────┴──────┬───────┴──────┬───────┘                │
   GDELT/RSS,           │              ▼              ▼              ▼                         │
   Bluesky/Reddit,      │        Pandera validation  →  quarantine bad rows                   │
   SEC EDGAR)           └──────────────┬───────────────────────────────────────────────────┘
                                       ▼
                        ┌──────────────────────────────────────────────┐
                        │  Self-hosted Postgres + pgvector (Docker)     │
                        │   • prices (partitioned, BRIN/pg_partman)     │
                        │   • macro_series   • sentiment  • forecasts   │
                        │   • news_articles (JSONB + tsvector + pgvector)│
                        │   nightly pg_dump → R2/B2 · LAN/Tailscale only │
                        └───────────────┬──────────────────────────────┘
                                        │
              ┌─────────────────────────┼──────────────────────────────┐
              ▼                         ▼                              ▼
    ML pipeline (Darts)         FastAPI backend                 DuckDB/Parquet
    • feature build (FinBERT    • custom endpoints + forecasts  • ATTACH Postgres +
      sentiment + macro +       • WebSocket/SSE live updates       read Parquet history
      technicals, lagged)       • Redis cache, ARQ workers      • feature engineering
                                                                • archives hot DB → small
    • walk-forward train               │
    • MLflow tracking                  ▼
    • batch predict → DB        ┌──────────────────────────────┐
                                │  UI: OpenBB Workspace /        │
                                │  Streamlit → (later) Next.js   │
                                │  charts · news · map · macro · │
                                │  forecasts                     │
                                └──────────────────────────────┘
```

---

## 5. Phased implementation plan

Estimates assume solo, part-time effort. Each phase ends in something usable.

### Phase 0 — Foundation & cleanup _(~1–2 weeks)_ — ✅ **DONE (2026-06-13)**
**Goal: a trustworthy base.**
- ✅ Fixed `requirements.txt` (added tensorflow, scikit-learn, matplotlib, pydantic-settings; pinned); split into `requirements.txt` + `requirements-dev.txt`; added `pyproject.toml`.
- ✅ Added `sklearn-env/`, `.env`, model artifacts to `.gitignore`.
- ✅ Restructured into a package: `src/market_intel/{config,data,models}/` with thin CLIs (`multi_feature_model.py`, `preprocess.py`).
- ✅ **Fixed the LSTM leakage** (`models/windowing.py`): split first → fit feature *and* target scalers on **train only** → window strictly causally within each split; shared helper across the fixed split and walk-forward (`TimeSeriesSplit`) paths; added early stopping. Leaky baseline test RMSE 30.32 → honest 12.55 on AAPL (walk-forward mean 13.68). Verified by a 3-lens adversarial review (no leakage found).
- ✅ Format-agnostic loader (`data/loaders.py`) handles single- *and* multi-ticker yfinance CSVs; fixed the triple-`__main__` bug in the fetcher.
- ✅ Added `.env` + `pydantic-settings` config (`config.py`); `.env.example`.
- ✅ `docker-compose.yml`: **`pgvector/pgvector:pg17`** + **Redis**, `restart: unless-stopped`, localhost-only, with `db/init/01_extensions.sql` enabling `vector`. Plain Postgres — **no TimescaleDB** (see §3.1). _(pg_cron deferred to Phase 1 — not in the pgvector image.)_
- ✅ Added `pytest` (30 tests incl. strict leakage regression tests for both the fixed split and walk-forward), `ruff`/`black` config, and a real README.

**Done:** the LSTM trains with honest (non-leaky) metrics; `pip install -r requirements.txt` reproduces the env; `docker compose up` provisions Postgres+pgvector+Redis. _Remaining for the user: spin up Docker on the always-on box and copy `.env.example` → `.env`._

### Phase 1 — Data ingestion backbone _(~2–4 weeks)_ — 🚧 **in progress**
**Goal: continuous, validated, multi-source ingestion into Postgres.**
- ✅ **Storage layer** (`storage/`): SQLAlchemy `Price` model (composite PK → idempotent upserts), engine/session helpers, prices repo. Portable Postgres/SQLite.
- ✅ **Market ingestor** (`ingest/`): CSV → **Pandera validation gate** → idempotent upsert. CLI `python src/ingest.py AAPL`. Verified end-to-end against real Postgres (pgvector/pg17 container): 2264 AAPL + 2264 MSFT rows, re-run stays idempotent.
- ✅ **Macro (FRED) ingestor**: fetch → parse (drops `.` missing values) → validate → idempotent upsert into a `macro_series` table. Verified against real Postgres. Isolated/injectable HTTP client so parsing is unit-tested without a key.
- ✅ **APScheduler worker** (`scheduler.py` + `src/worker.py`): registers market + FRED + GDELT ingestion jobs (per-item errors logged, never crash the loop); registration unit-tested.
- ✅ **GDELT news ingestor** (the "edge-of-the-world" core): free/no-key DOC 2.0 ArtList → parse (validation gate + in-batch dedup, keeps first/richest record) → idempotent upsert into `news_articles`. `news_articles` schema with `raw` **JSONB** (portable JSON on SQLite). 30-min scheduled job; one-shot CLI `python src/ingest_news.py "<query>"`. Verified on real Postgres (`raw` is genuinely `jsonb`, idempotent); live API call confirmed reachable.
- ✅ **News search layer**: keyword **FTS** (Postgres `to_tsvector`/`websearch_to_tsquery`, SQLite `LIKE` fallback) + **semantic vector search** (pgvector `embedding` column + cosine `<=>`; in-Python cosine on SQLite). Pluggable `Embedder` (dependency-free hashing baseline now; `sentence-transformers` is a 384-dim drop-in upgrade). GIN + HNSW indexes via `ensure_search_indexes`; embeddings backfilled in the GDELT job. CLI `python src/search_news.py "<q>" [--semantic]`. Verified on real Postgres (vector column, HNSW+GIN indexes, cosine ranking).
- ✅ **SEC EDGAR filings ingestor**: ticker→CIK resolution + submissions API → parse `filings.recent` (validation gate) → idempotent upsert into a `filings` table. Daily scheduled job; CLI `python src/ingest_filings.py AAPL`. **Verified live end-to-end** (1000 real AAPL filings ingested). Needs a descriptive `SEC_USER_AGENT`.
- ✅ **Live yfinance→DB path**: `ingest_yfinance` (injectable downloader) normalizes flat/MultiIndex output → validate → upsert; `python src/ingest.py AAPL --live`. (Yahoo can rate-limit; empty results handled gracefully.)
- ✅ **Nightly backup**: `scripts/backup.sh` (`pg_dump -Fc` → Cloudflare R2 / Backblaze B2 via rclone, retention prune) + cron wiring docs.
- ⬜ Optional extras: DBnomics, curated FreshRSS; range-partitioning (BRIN now → `pg_partman`+`pg_cron` when large) — a scaling optimization, deferred until tables are big.
- _Tests: 82 passing (EDGAR incl. idempotency/ragged-array edge cases; live-yfinance normalization incl. lowercase ticker, reversed MultiIndex, tz-aware index, NaN handling — hardened via adversarial review); ruff/black clean._

**Phase 1 is functionally complete** — market (CSV + live), macro (FRED), global news (GDELT) + keyword/semantic search, and filings (EDGAR) all ingest idempotently on a schedule into Postgres with validation gates. Remaining items are optional/scaling.

**Done when:** scheduled jobs keep prices, macro, and global news flowing into Postgres unattended, with validation gates and off-box backups.

### Phase 2 — Dashboard MVP _(~3–4 weeks)_ — 🚧 **in progress**
**Goal: see it.**
- ✅ **FastAPI JSON API** (`market_intel/api`): frontend-agnostic endpoints over everything ingested — `/api/prices/{symbol}`, `/api/indicators/{symbol}`, `/api/macro/{id}`, `/api/news/recent`, `/api/news/search` (keyword + semantic), `/api/filings/{ticker}`, `/api/health`. Test-injectable session factory; tested via FastAPI TestClient. CLI `python src/serve.py`.
- ✅ **Self-contained dark terminal dashboard** (`api/static/index.html`, no build step): candlestick price chart (TradingView Lightweight Charts), live GDELT news feed with keyword/semantic search, macro line chart, filings list. Verified end-to-end against real Postgres (500 price bars + 1000 EDGAR filings served).
- ⬜ **World map of GDELT events** (Leaflet) — uses article `source_country`/geo.
- ✅ **Technical indicators on the price chart**: SMA 20/50, EMA, **RSI** (Wilder's smoothing), **MACD**, **Bollinger Bands** computed server-side and **causally** (`market_intel/indicators.py` — `rolling`/`ewm` only, no lookahead) → `/api/indicators/{symbol}` (date-aligned, JSON-safe NaN→null) → dashboard overlays (MA + Bollinger), a toggleable **RSI** sub-band, and a latest-value readout (MA20 / RSI / MACD). Indicators load independently of the price fetch and degrade gracefully if unavailable. Tests cover the indicator math (known values, warm-up, causality, all-gains/all-losses edges) + the endpoint.
- ⬜ Ticker-level sentiment on the news feed — **deferred**: depends on FinBERT scoring (a Phase-3 deliverable) or a keyed sentiment API (Marketaux/Finnhub); not self-contained at this stage.
- ⬜ Live updates via **FastAPI WebSockets/SSE** + Redis caching (currently fetch-on-load).
- ⬜ (Optional) OpenBB Workspace widgets pointed at the same API.

**Done when:** one screen shows live markets + news + macro + a world event map, auto-refreshing.

### Phase 3 — Clean ML forecasting layer _(~3–4 weeks)_
**Goal: trustworthy, displayed predictions.**
- Port the LSTM into **Darts**; enforce split-before-window, train-only scaling, walk-forward validation.
- **FinBERT** sentiment features: score news bodies, aggregate to daily per-asset scores, **lag** them; fuse with technicals + lagged macro into one multivariate matrix.
- **MLflow** tracking + model registry; **DVC**/Parquet snapshots for reproducible datasets.
- Scheduled **batch prediction** → write forecasts to DB → render in the dashboard with confidence bands.

**Done when:** forecasts (with honest backtests) are produced on a schedule and shown next to actuals.

### Phase 4 — Intelligence & scale _(months 4–6+)_
**Goal: from dashboard to signal engine.**
- Semantic news search (pgvector) + entity/event extraction; LLM summaries on top-N flagged stories.
- **Event → asset linkage**: map a remote headline to the tickers/sectors/regions it may move; volatility-regime awareness.
- Alerting (anomaly/sentiment-spike notifications).
- Social firehose (Bluesky Jetstream + Reddit) for early signals.
- Model upgrades: benchmark **N-HiTS** / **TFT** (quantile forecasts, future covariates) via Darts/NeuralForecast.
- Scale the stack as needed (Prefect, ClickHouse/QuestDB, Qdrant, BentoML, Next.js front end, Hetzner VPS).

**Done when:** the system proactively surfaces "this faint event may move that asset," with model and infra that scale.

---

## 6. Next steps

**Phase 0 is complete** (see above). Remaining setup for the user:
1. Provision the **Oracle Cloud Always Free** VM (convert to PAYG, pick an A1-capacity region), install Docker + Tailscale, then `docker compose up -d`, `cp .env.example .env`, set `DB_PASSWORD`, and wire the nightly `pg_dump` → R2/B2 backup. _(Or run the same `docker compose up` locally to start.)_
2. Get **free API keys** for Phase 1: FRED, Finnhub, (optional) Marketaux. Try the **GDELT DOC API** with a `curl` query — no key needed. (Tiingo is optional/paid — skip for now.)

**Then Phase 1 — ingestion backbone:** uncomment the Phase-1 deps in `requirements.txt` (SQLAlchemy, psycopg, APScheduler, pandera…), define the `prices`/`macro_series`/`news_articles` schema, and stand up the first scheduled ingestors (market + FRED + GDELT) writing into Postgres with Pandera validation.

## 7. Key risks & notes
- ✅ **ML leakage** (was the #1 correctness risk) — fixed in Phase 0 and locked by `tests/test_windowing.py`.
- **Self-hosting = you own DR & uptime** — the nightly `pg_dump` → R2/B2 is non-negotiable; keep the DB LAN-only/Tailscale (never expose port 5432); disable host sleep.
- **No TimescaleDB** (see §3.1) — use native partitioning + `pg_partman` + DuckDB/Parquet; archive history to Parquet to keep the hot DB small. (It's *available* self-hosted but unneeded at this scale.)
- **Vendor terms**: yfinance is unofficial/personal-use-only; GDELT explicitly allows redistribution; verify each API's storage/redistribution clause before persisting.
- **Pricing drifts** — re-check free-tier limits (Alpha Vantage tightened to ~25/day; X API is now cost-prohibitive; IEX Cloud shut down Aug 2024; Polygon rebranded to "Massive").
- **Scope discipline** — resist building the scaling stack early; the lean stack is deliberately small so you ship.
```
