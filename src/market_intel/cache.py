"""Optional Redis cache for the read endpoints — cache-aside with graceful fallback.

The dashboard must never break because the cache is unavailable, so degradation is
built in at two levels (mirrors the SQLite/embeddings fallbacks elsewhere in the repo):

* **Startup** — :func:`build_cache` imports ``redis`` and pings once. On ``ImportError``
  or any connection error it returns a :class:`NullCache` (caching simply off).
* **Runtime** — every Redis call is wrapped; a live error is swallowed and treated as a
  cache miss / dropped write, so reads fall through to Postgres.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, Protocol

log = logging.getLogger(__name__)


class Cache(Protocol):
    """Minimal JSON cache interface (duck-typed; ``MemoryCache`` is the test double)."""

    enabled: bool

    def get_json(self, key: str) -> Any | None: ...
    def set_json(self, key: str, value: Any, ttl: int) -> None: ...


class NullCache:
    """No-op cache: every get is a miss, every set is dropped."""

    enabled = False

    def get_json(self, key: str) -> Any | None:
        return None

    def set_json(self, key: str, value: Any, ttl: int) -> None:
        pass


class MemoryCache:
    """In-process dict cache for tests (no TTL expiry — fine for a short test run)."""

    enabled = True

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get_json(self, key: str) -> Any | None:
        raw = self._store.get(key)
        return None if raw is None else json.loads(raw)

    def set_json(self, key: str, value: Any, ttl: int) -> None:
        self._store[key] = json.dumps(value)


class RedisCache:
    """JSON wrapper over a redis client; all ops degrade to miss/no-op on error."""

    enabled = True

    def __init__(self, client: Any) -> None:
        self._client = client

    def get_json(self, key: str) -> Any | None:
        try:
            raw = self._client.get(key)
        except Exception:  # pragma: no cover - redis runtime/network error
            log.warning("cache get failed for %s", key, exc_info=True)
            return None
        return None if raw is None else json.loads(raw)

    def set_json(self, key: str, value: Any, ttl: int) -> None:
        try:
            self._client.set(key, json.dumps(value), ex=ttl)
        except Exception:  # pragma: no cover - redis runtime/network error
            log.warning("cache set failed for %s", key, exc_info=True)


def build_cache(url: str | None = None, *, enabled: bool = True) -> Cache:
    """Return a working :class:`RedisCache`, or :class:`NullCache` if Redis is unavailable.

    Pings once so an unreachable server downgrades cleanly at startup rather than raising
    on the first request.
    """
    if not enabled:
        return NullCache()

    from market_intel.config import settings

    url = url or settings.redis_url
    try:
        import redis

        client = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=1)
        client.ping()
    except Exception as exc:  # ImportError or any connection error
        log.warning("Redis unavailable (%s: %s) — caching disabled", url, exc)
        return NullCache()
    return RedisCache(client)


def cached(cache: Cache, key: str, ttl: int, producer: Callable[[], Any]) -> Any:
    """Cache-aside: return the cached value for ``key`` or compute, store, and return it.

    ``producer`` must return a JSON-serializable value. A ``None`` result is not cached
    (it is indistinguishable from a miss), which is fine — the endpoints return lists.
    """
    hit = cache.get_json(key)
    if hit is not None:
        return hit
    value = producer()
    if value is not None:
        cache.set_json(key, value, ttl)
    return value
