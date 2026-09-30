"""Central configuration. Everything is env-driven; no secrets in code."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Providers: "stub" | "openai_compat" ; "stub" | "tavily"
    llm_provider: str = Field(default="stub")
    llm_base_url: str = Field(default="http://localhost:11434/v1")
    llm_api_key: str = Field(default="")
    llm_model: str = Field(default="qwen2.5:1.5b")

    search_provider: str = Field(default="stub")
    tavily_api_key: str = Field(default="")

    redis_url: str = Field(default="redis://localhost:6379/0")

    # Per-run budgets
    max_tokens_per_run: int = Field(default=60_000, ge=1)
    max_searches_per_run: int = Field(default=12, ge=1)
    max_subquestions_per_run: int = Field(default=5, ge=1)
    max_revisions_per_run: int = Field(default=2, ge=0)
    wall_clock_timeout_sec: int = Field(default=600, ge=10)

    # Cost model: USD per 1M tokens. Estimates only — real spend depends on the
    # configured provider's pricing page.
    price_input_per_1m_usd: float = Field(default=0.15, ge=0)
    price_output_per_1m_usd: float = Field(default=0.60, ge=0)

    log_level: str = Field(default="INFO")
    step_delay_sec: float = Field(default=0.0, ge=0.0)  # test hook, see .env.example


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
