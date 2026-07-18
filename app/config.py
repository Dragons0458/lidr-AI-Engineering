from functools import lru_cache
from pathlib import Path
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
    # --- Session 7: chunking strategies + runtime model catalog ---
    AVAILABLE_MODELS: list[str] = Field(
        default_factory=lambda: [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-5",
            "gpt-5-mini",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5",
        ]
    )
    PROPOSITIONAL_CHUNKER_MODEL: str = "gpt-4o-mini"
    CONTEXTUAL_CHUNKER_MODEL: str = "claude-sonnet-4-5"
    # --- Session 9: RAG end-to-end (transcript → grounded estimate) ---
    REFORMULATION_MODEL: str = "gpt-5-mini"
    GENERATION_MODEL: str = "gpt-5"
    GENERATION_REASONING_EFFORT: Literal["minimal", "low", "medium", "high"] = "high"
    GENERATION_MAX_TOKENS: int = 64000
    RETRIEVAL_TOP_K: int = 10
    RETRIEVAL_DISTANCE_THRESHOLD: float = 0.6
    MAX_CONTEXT_TOKENS: int = 16384
    IDEMPOTENCY_TTL: int = 86400
    RETRIEVAL_API_KEY: str | None = None
    ESTIMATE_API_KEY: str | None = None
    # --- Session 10: hybrid search + cross-encoder reranking ---
    RETRIEVAL_SEARCH_MODE: Literal["vector", "hybrid"] = "vector"
    RERANKER_ENABLED: bool = False
    RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    RETRIEVAL_RECALL_TOP_K: int = 50
    RERANK_TOP_N: int = 5
    RRF_K: int = 60
    # --- Session 10 live: advanced retrieval pipeline ---
    RETRIEVAL_ROUTING_ENABLED: bool = True
    QUERY_TRANSFORM_ENABLED: bool = True
    TEMPORAL_DECAY_ENABLED: bool = False
    ROUTER_MODEL: str = "gpt-4o-mini"
    QUERY_TRANSFORM_MODEL: str = "gpt-4o-mini"
    TEMPORAL_DECAY_HALF_LIFE_DAYS: int = 900
    QUERY_MAX_SUBQUERIES: int = 4
    ROUTER_MAX_TARGETS: int = 3
    TASK_HOURS_TOP_K: int = 5
    TASK_HOURS_DISTANCE_THRESHOLD: float = 0.45
    # --- Session 11: augmentation + synthesis + hallucination gate ---
    AUGMENTATION_ENABLED: bool = True
    AUGMENTATION_COMPRESS: bool = True
    AUGMENTATION_REORDER: bool = True
    AUGMENTATION_MODEL: str = "gpt-5-mini"
    SYNTHESIS_ENABLED: bool = True
    SYNTHESIS_CONTRADICTION_THRESHOLD: float = 0.35
    HALLUCINATION_GATE_ENABLED: bool = True
    HALLUCINATION_JUDGE_MODEL: str = "gpt-5-mini"
    HALLUCINATION_NUMERIC_TOLERANCE: float = 0.5
    # --- Session 12: hand-written estimation agent (Responses API) ---
    AGENT_MODEL: str = "gpt-5"
    AGENT_REASONING_EFFORT: Literal["minimal", "low", "medium", "high"] = "medium"
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_SEARCH_TOP_K: int = 5
    AGENT_SEARCH_DISTANCE_THRESHOLD: float = 0.45
    AGENT_RECOVERY_RELIABILITY_THRESHOLD: float = Field(default=0.35, ge=0, le=1)
    # --- Session 13: LangGraph orchestration + Logfire ---
    LOGFIRE_TOKEN: str | None = None
    LOGFIRE_SERVICE_NAME: str = "estimador-cag"
    LANGGRAPH_ENABLED: bool = True
    GRAPH_CLASSIFIER_MODEL: str = "gpt-5-mini"
    GRAPH_ANALYSIS_MODEL: str = "gpt-5"
    GRAPH_PROPOSAL_MODEL: str = "gpt-5"
    GRAPH_PROPOSAL_ENABLED: bool = True
    GRAPH_PERSONAS_ENABLED: bool = True
    GRAPH_STRUCTURE_EFFORT_BY_COMPLEXITY: dict[str, str] = Field(
        default_factory=lambda: {"low": "low", "medium": "medium", "high": "high"}
    )
    GRAPH_ACTIVITY_TTL: int = 3600
    # --- Session 6: ingestion + persistence + PII ---
    DATABASE_URL: str = (
        "postgresql+psycopg://estimator:estimator@localhost:5433/estimator"
    )
    CATALOG_PATH: Path = Path("data/catalog/catalog.yaml")
    INGESTION_DATA_ROOT: Path = Path("data/seed")
    PRESIDIO_SPACY_MODEL: str = "es_core_news_md"
    PSEUDONYM_FAKER_LOCALE: str = "es_ES"
    PSEUDONYM_HASH_SALT: str = "change-me-in-prod"
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
