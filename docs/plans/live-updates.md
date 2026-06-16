# Plan — Live updates (SSE) + Redis caching (Phase 2)

_Roadmap item (§5, Phase 2): "Live updates via **FastAPI WebSockets/SSE** + **Redis
caching** (currently fetch-on-load)." This is the last functional piece of the Phase 2
"Done when … **auto-refreshing**" criterion (the world map ships separately in its own PR)._

## Goal

Turn the fetch-on-load dashboard into a **live, auto-refreshing** terminal:

1. **Server-Sent Events (SSE)** stream that pushes the global news feed as new GDELT
   articles land, and ticks the browser to refresh the symbol panels — without a reload.
2. **Redis cache** in front of the read endpoints so page loads, panel refreshes, and
   every SSE client share one short-lived cached result instead of hammering Postgres.

Both follow the existing repo → API → UI pattern and the project's "boring, free,
self-hosted, graceful-degradation" principles (cf. the SQLite/embeddings fallbacks).

## Why SSE (not WebSockets), why polling-push (not pub/sub)

- The dashboard data flow is **one-way** (server → browser). SSE is the right tool:
  plain HTTP, native `EventSource` with **built-in auto-reconnect**, and far less code
  than a WebSocket handshake/lifecycle. (Verified against current FastAPI/Starlette
  guidance — SSE is recommended over raw streaming for structured, reconnecting feeds.)
- Implemented with Starlette's `StreamingResponse(media_type="text/event-stream")` —
  **no new framework dependency** (no `sse-starlette`), matching the repo's minimalism.
- The ingestion worker (`src/worker.py`) and the API (`src/serve.py`) are **separate
  processes**. True worker→API push needs a Redis pub/sub bridge (roadmap: "Redis
  pub/sub for fan-out later"). For the MVP the SSE generator instead **polls the DB on a
  short interval** and emits only on change. This works even when the API runs alone, and
  the Redis cache keeps the poll cheap. Pub/sub fan-out remains a clean later upgrade.

## Design

### 1. `cache.py` — Redis cache-aside with graceful fallback
- `build_cache(url)` lazily constructs a `RedisCache` (imports `redis`, pings once); on
  **ImportError / connection failure → `NullCache`** (no-op). Caching is always optional.
- `Cache.get_json(key)` / `set_json(key, obj, ttl)`. Every Redis call is wrapped so a
  **runtime** Redis error degrades to a cache miss (read falls through to Postgres) — the
  dashboard never breaks because Redis is down. Best-practice graceful fallback.
- `MemoryCache` (dict-backed) for tests — exercises the cache-aside path without a server.
- `cached(cache, key, ttl, producer)` helper: get-or-compute-and-store.

### 2. Cache the read endpoints (`api/app.py`)
- Apply `cached(...)` to the **load** endpoints: `prices`, `macro`, `news/recent`,
  `filings`, keyed by path + params, TTL = `settings.cache_ttl_seconds` (default 15s).
- Search endpoints stay uncached (user-driven, high-cardinality — low hit rate).
- `create_app(session_factory=None, cache=None)` — inject a `MemoryCache` in tests.

### 3. SSE endpoint `GET /api/stream`
- Async generator; each `settings.stream_poll_seconds` (default 10s):
  - Read cheap news state `(count, max(seen_date))` off the DB in a thread
    (`asyncio.to_thread`, so the sync session never blocks the event loop).
  - On **first tick or change** → emit `event: news` with the recent article list
    (same shape as `/api/news/recent`), so the client just re-renders the feed.
  - Always emit `event: tick` `data:{ts,count}` — doubles as the keepalive heartbeat
    (also a `:` comment line for proxies) and lets the client refresh symbol panels.
  - Break when `await request.is_disconnected()`; catch `CancelledError` for clean exit.

### 4. Frontend (`api/static/index.html`)
- Open `EventSource('/api/stream')`. On `news` → `renderNews(...)`; on `tick` →
  refresh the symbol panels (prices/filings/macro) at most once/60s (cheap, cached).
- A **● LIVE** status dot in the header: green on `onopen`, muted "reconnecting…" on
  `onerror` (EventSource reconnects itself). Initial fetch-on-load is kept as the
  first paint; SSE takes over for updates.

### 5. Config / deps
- `config.py`: `cache_ttl_seconds=15`, `stream_poll_seconds=10.0`, `cache_enabled=True`.
- `requirements.txt`: add `redis==5.3.1` (was the commented Phase-1 placeholder).
- `.env.example`: note the two knobs.

## Testing
- `test_cache.py`: `MemoryCache` get/set/TTL-miss; `NullCache` no-ops; `build_cache`
  returns `NullCache` on a bad URL (graceful); `cached()` computes once then serves cached.
- `test_api.py`: endpoints serve identical data with a `MemoryCache` injected (cache hit
  path); a second call doesn't re-query (producer called once).
- `test_stream.py`: `GET /api/stream` returns `text/event-stream`; the generator emits an
  initial `news` event with the seeded articles and a `tick` event (drive the async
  generator directly with a tiny poll interval and a disconnect after the first cycle).

## Out of scope (future)
- Redis **pub/sub** worker→API fan-out (true push); WebSockets; per-symbol price streaming.
