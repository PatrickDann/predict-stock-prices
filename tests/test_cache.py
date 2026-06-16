"""Cache module: the cache-aside helper plus MemoryCache / NullCache / graceful build."""

from market_intel.cache import MemoryCache, NullCache, RedisCache, build_cache, cached


def test_memory_cache_roundtrip():
    c = MemoryCache()
    assert c.get_json("k") is None
    c.set_json("k", {"a": 1}, ttl=10)
    assert c.get_json("k") == {"a": 1}


def test_null_cache_is_noop():
    c = NullCache()
    c.set_json("k", [1, 2, 3], ttl=10)
    assert c.get_json("k") is None
    assert c.enabled is False


def test_cached_computes_once_then_serves_cache():
    c = MemoryCache()
    calls = {"n": 0}

    def producer():
        calls["n"] += 1
        return [1, 2, 3]

    assert cached(c, "key", 10, producer) == [1, 2, 3]
    assert cached(c, "key", 10, producer) == [1, 2, 3]
    assert calls["n"] == 1  # second call served from cache


def test_cached_does_not_cache_none():
    c = MemoryCache()
    calls = {"n": 0}

    def producer():
        calls["n"] += 1
        return None

    assert cached(c, "key", 10, producer) is None
    assert cached(c, "key", 10, producer) is None
    assert calls["n"] == 2  # None is indistinguishable from a miss → recomputed


def test_cached_with_null_cache_recomputes_every_time():
    c = NullCache()
    calls = {"n": 0}

    def producer():
        calls["n"] += 1
        return [9]

    assert cached(c, "k", 10, producer) == [9]
    assert cached(c, "k", 10, producer) == [9]
    assert calls["n"] == 2  # nothing is stored → recomputed


def test_build_cache_disabled_returns_nullcache():
    assert isinstance(build_cache(enabled=False), NullCache)


def test_build_cache_unreachable_redis_degrades_to_nullcache():
    # Connection-refused (unused port) must downgrade cleanly, not raise.
    assert isinstance(build_cache("redis://127.0.0.1:1/0", enabled=True), NullCache)


class _BoomClient:
    """A redis-like client whose every op raises — exercises runtime fallback."""

    def get(self, key):
        raise RuntimeError("redis down")

    def set(self, key, value, ex=None):
        raise RuntimeError("redis down")


def test_redis_cache_swallows_runtime_errors():
    c = RedisCache(_BoomClient())
    assert c.get_json("k") is None  # error treated as a miss
    c.set_json("k", [1], ttl=5)  # error swallowed, no raise
