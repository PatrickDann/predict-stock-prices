-- Runs once on first DB initialization (mounted into docker-entrypoint-initdb.d).
-- Enables the extensions the project relies on. See docs/VISION_AND_ROADMAP.md §3.1.

CREATE EXTENSION IF NOT EXISTS vector;   -- pgvector: embeddings / semantic news search

-- pg_cron (in-DB scheduling for partition maintenance / rollups) is NOT in the
-- pgvector/pgvector image. Add it in Phase 1 via an image that bundles it plus
-- `shared_preload_libraries = 'pg_cron'`. Until then, APScheduler in the Python
-- worker handles scheduling.
--
-- TimescaleDB is intentionally NOT used: time-series uses native range
-- partitioning (BRIN now -> pg_partman later) + Parquet/DuckDB for ML history.
