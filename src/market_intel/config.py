"""Typed application configuration.

Values come from environment variables and an optional `.env` file (see
`.env.example`). Import the singleton `settings` or call `get_settings()`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database (self-hosted Postgres + pgvector) ---
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "market_intel"
    db_user: str = "market_intel"
    db_password: str = "change_me"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Data source API keys (optional in Phase 0) ---
    fred_api_key: str | None = None
    finnhub_api_key: str | None = None
    tiingo_api_key: str | None = None
    marketaux_api_key: str | None = None

    # --- Paths ---
    data_dir: Path = Field(default=REPO_ROOT / "data")
    models_dir: Path = Field(default=REPO_ROOT / "models")

    @property
    def database_url(self) -> str:
        """SQLAlchemy/psycopg connection URL (credentials percent-encoded)."""
        user = quote_plus(self.db_user)
        password = quote_plus(self.db_password)
        return (
            f"postgresql+psycopg://{user}:{password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
