# Plan — DBnomics global macro ingestor (Phase 1)

_Status: implemented on `feat/phase-1-dbnomics-macro`. Branch base: `main`._

## Why this feature

The roadmap's macro layer is **US-only** today: the FRED ingestor
(`ingest/macro.py`) covers ~95% of US macro but nothing global. The §3 stack
explicitly pairs FRED with **DBnomics** — a free aggregator of World Bank, IMF,
OECD, Eurostat, AMECO and ~80 other providers — to reach the "ingests globally"
North Star (§2). It is listed under Phase 1 "Optional extras". All the core
Phase 2 dashboard items are already in flight on other branches (world map #9,
sentiment #12, live updates #11, technical indicators merged), so DBnomics is
the highest-value *un-branched* roadmap item and is fully self-contained.

## Research notes (verified against the live API, June 2026)

- **No API key, free, redistribution-friendly.** Single endpoint:
  `GET https://api.db.nomics.world/v22/series?series_ids=<id>&observations=1`.
- A series is identified by the triple **`PROVIDER/DATASET/SERIES`**, e.g.
  `Eurostat/namq_10_gdp/Q.CLV10_MEUR.SCA.B1GQ.DE`,
  `IMF/WEO:latest/USA.NGDP_RPCH`, `WB/WDI/...`.
- Response shape: `series.docs[0]` holds parallel arrays:
  - `period` — period labels (`"1991-Q1"`, `"2020"`, `"2020-01-01"`).
  - `period_start_day` — the **ISO start date** of each period (`"1991-01-01"`).
    Preferred for storage: already a clean date, no period-format parsing.
  - `value` — observations; **missing values are the string `"NA"`** (DBnomics
    normalizes every provider's missing marker to `"NA"`).
- DBnomics docs recommend a cache layer / "focused" queries (one series at a
  time), not bulk downloads. We poll one series per call on a daily cron.

## Design — mirror the FRED ingestor exactly

`ingest/macro.py` already factors the generic tail (`ingest_macro_frame` →
`validate_macro` → `upsert_macro`) out of the FRED-specific head. DBnomics reuses
that tail verbatim and only adds a new fetch+parse head, in a sibling module.

**`src/market_intel/ingest/dbnomics.py`** (mirrors `macro.py`):
- `fetch_dbnomics_series(series_id, *, url, get, timeout)` — GET with
  `series_ids=<id>&observations=1`; `get` is injectable (no network in tests).
- `parse_dbnomics_series(payload)` — pull `series.docs[0]`, prefer
  `period_start_day` (fall back to `period`), pair with `value`,
  `pd.to_numeric(errors="coerce")` turns `"NA"` → NaN, drop NaN, sort. Empty /
  malformed docs → empty frame (same as FRED's empty-payload behaviour).
- `ingest_dbnomics(session, series_id, *, get)` — fetch → parse →
  `ingest_macro_frame(session, df, series_id, source="DBnomics")`.

**Storage:** reuse the `macro_series` table. DBnomics ids (with `/`, `:`, `.`)
are longer than FRED's; the widest real ids (~45 chars) exceed the current
`String(40)` PK column, so widen `MacroSeries.series_id` to `String(128)`.
The project has no migration framework (it uses `create_all`); widening the
model is the correct, low-risk change (SQLite ignores length; fresh Postgres
picks up 128). **Caveat:** `create_all` does not `ALTER` an *already-existing*
`macro_series` table, so an operator with a pre-Phase-1 Postgres deployment must
run `ALTER TABLE macro_series ALTER COLUMN series_id TYPE VARCHAR(128);` once
before ingesting long DBnomics ids (FRED ids ≤40 chars are unaffected). The
scheduled job's per-item `try/except` turns a too-long id into a logged skip,
not a crash; the one-shot CLI would surface the DB error directly.

**Scheduling:** `ingest_dbnomics_job(series_ids, session_factory)` in
`scheduler.py` (per-item try/except like every other job), registered by
`build_scheduler(..., dbnomics_series=())` on a daily cron. No key gate (unlike
FRED) since DBnomics is keyless.

**CLIs:**
- One-shot `src/ingest_dbnomics.py "<PROVIDER/DATASET/SERIES>" ...` — mirrors
  `ingest_news.py` / `ingest_filings.py` for ad-hoc backfills (keyless, handy).
- `--dbnomics` flag on `src/worker.py`.

## Out of scope (deliberate)

- **No API changes.** The generic `/api/macro/{series_id}` endpoint is Phase 2
  and owned by in-flight branches; DBnomics ids contain `/` (path-routing) and
  the endpoint `.upper()`s ids (DBnomics codes are case-sensitive), so wiring
  them into that path is its own change. DBnomics rows are queryable from the DB
  now; the dashboard hook can follow once the Phase 2 API PRs land.
- No new dependencies (uses the existing `requests` + `pandas` stack).

## Test plan (`tests/test_dbnomics.py`, mirrors `test_fred.py`)

- parse: `"NA"` dropped, out-of-order sorted, uses `period_start_day`.
- parse: empty / missing `docs` → empty frame.
- fetch: builds the correct request (`series_ids`, `observations=1`); raises on
  HTTP error.
- end-to-end ingest into SQLite, source recorded as `DBnomics`, idempotent.

## Verification gates

`/simplify` → `/code-review` → `/security-review` → full `pytest` green +
`ruff`/`black` clean. Roadmap updated in the same branch. PR to `main`, no merge.
