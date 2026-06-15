"""Configuration loads with sane defaults and builds a DB URL."""

from market_intel.config import Settings


def test_defaults_and_database_url():
    s = Settings(_env_file=None)
    assert s.db_port == 5432
    assert s.db_name == "market_intel"
    url = s.database_url
    assert url.startswith("postgresql+psycopg://")
    assert "@localhost:5432/market_intel" in url


def test_env_override(monkeypatch):
    monkeypatch.setenv("DB_HOST", "db.internal")
    monkeypatch.setenv("DB_PASSWORD", "secret")
    s = Settings(_env_file=None)
    assert s.db_host == "db.internal"
    assert "secret@db.internal" in s.database_url
