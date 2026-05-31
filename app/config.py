from functools import lru_cache
from typing import Any, Literal

from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    LLM_PROVIDER: Literal["openai", "anthropic", "google"] = "openai"
    PRIMARY_MODEL: str = "gpt-4o-mini"
    FALLBACK_MODEL: str | None = None
    LLM_TIMEOUT: int = 30
    LLM_RETRIES: int = 2
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 86400
    CACHE_ENABLED: bool = True
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_THRESHOLD: float = 0.88
    SEMANTIC_CACHE_TTL: int = 86400
    SEMANTIC_CACHE_LOG_ONLY: bool = False
    INPUT_GUARDRAILS_ENABLED: bool = True
    OUTPUT_GUARDRAILS_ENABLED: bool = True
    CONVERSATION_MAX_TURNS: int = Field(default=6, ge=0)
    # --- Session 5: tier ---
    TIER_RESOLUTION_ENABLED: bool = False
    # --- Session 5: memory compression ---
    MEMORY_COMPRESSION_ENABLED: bool = False
    ANCHOR_DETECTION_MODE: Literal["heuristic", "llm"] = "heuristic"
    COMPRESSION_MODEL: str | None = None
    # --- Session 5: Actor-Critic-Boss ---
    CRITIC_MODEL: str | None = None
    BOSS_MAX_ITERATIONS: int = Field(default=3, ge=1, le=5)
    APP_ENV: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"
    CORS_ALLOWED_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])

    @model_validator(mode="before")
    @classmethod
    def map_legacy_llm_model(cls, data: Any) -> Any:
        """Map removed LLM_MODEL env var to PRIMARY_MODEL for existing .env files."""
        if not isinstance(data, dict):
            return data
        primary = data.get("PRIMARY_MODEL") or data.get("primary_model")
        legacy = data.get("LLM_MODEL") or data.get("llm_model")
        if not primary and legacy:
            data = {**data, "PRIMARY_MODEL": legacy}
        return data

    @model_validator(mode="after")
    def validate_api_key_for_provider(self) -> "Settings":
        """Ensure the API key for the selected LLM provider is present."""
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'")
        if self.LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER is 'anthropic'"
            )
        if self.LLM_PROVIDER == "google" and not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER is 'google'")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings (singleton)."""
    return Settings()
