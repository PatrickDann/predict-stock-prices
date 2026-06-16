"""SSE live-updates generator: emits a `news` event on change + a `tick` each cycle.

Drives ``news_stream`` directly with an injected ``is_disconnected`` so the test is
deterministic and fast (no TestClient streaming, which can't propagate disconnect to an
infinite SSE generator).
"""

import asyncio
import json

import pytest
from sqlalchemy.pool import StaticPool

from market_intel.api.app import news_stream
from market_intel.storage.db import init_db, make_engine, make_session_factory
from market_intel.storage.news_repo import upsert_articles


@pytest.fixture
def sf():
    # In-memory SQLite shared across threads: news_stream reads via asyncio.to_thread.
    engine = make_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    init_db(engine)
    factory = make_session_factory(engine)
    with factory() as s:
        upsert_articles(
            s,
            [
                {"url_hash": "h1", "url": "u1", "title": "Remote port blast"},
                {"url_hash": "h2", "url": "u2", "title": "Rate decision"},
            ],
        )
    return factory


def _stop_after(n: int):
    """is_disconnected() that returns False n times, then True (ends the stream)."""
    calls = {"n": 0}

    def is_disconnected():
        calls["n"] += 1
        return calls["n"] > n

    return is_disconnected


async def _collect(sf, cycles: int) -> list[str]:
    frames = []
    async for frame in news_stream(sf, 0.0, _stop_after(cycles)):
        frames.append(frame)
    return frames


def _parse(frames: list[str]) -> list[tuple[str, str]]:
    """Flatten SSE frames into (event, data) pairs."""
    events = []
    for frame in frames:
        lines = frame.strip().splitlines()
        event = next(line[len("event: ") :] for line in lines if line.startswith("event: "))
        data = next(line[len("data: ") :] for line in lines if line.startswith("data: "))
        events.append((event, data))
    return events


def test_stream_emits_news_then_tick(sf):
    events = _parse(asyncio.run(_collect(sf, cycles=2)))
    kinds = [e for e, _ in events]
    # First cycle: news (initial) + tick; second cycle: tick only (no change).
    assert kinds == ["news", "tick", "tick"]


def test_stream_news_payload_carries_articles(sf):
    events = _parse(asyncio.run(_collect(sf, cycles=1)))
    news = [json.loads(data) for kind, data in events if kind == "news"]
    assert news and {a["title"] for a in news[0]} == {"Remote port blast", "Rate decision"}


def test_stream_stops_when_disconnected(sf):
    # is_disconnected True immediately → no frames at all.
    assert asyncio.run(_collect(sf, cycles=0)) == []
