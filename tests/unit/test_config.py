from pathlib import Path

from app.config import Settings


def test_session5_defaults_are_off(monkeypatch) -> None:
    for env_name in (
        "TIER_RESOLUTION_ENABLED",
        "MEMORY_COMPRESSION_ENABLED",
        "COMPRESSION_MODEL",
        "CRITIC_MODEL",
    ):
        monkeypatch.delenv(env_name, raising=False)
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="test",
        LLM_PROVIDER="openai",
    )
    assert settings.TIER_RESOLUTION_ENABLED is False
    assert settings.MEMORY_COMPRESSION_ENABLED is False
    assert settings.ANCHOR_DETECTION_MODE == "heuristic"
    assert settings.COMPRESSION_MODEL is None
    assert settings.CRITIC_MODEL is None
    assert settings.BOSS_MAX_ITERATIONS == 3


def test_session6_defaults() -> None:
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="test",
        LLM_PROVIDER="openai",
    )
    assert "postgresql" in settings.DATABASE_URL
    assert settings.CATALOG_PATH.name == "catalog.yaml"
    assert settings.INGESTION_DATA_ROOT == Path("data/seed")
    assert settings.PRESIDIO_SPACY_MODEL == "es_core_news_md"
    assert settings.PSEUDONYM_HASH_SALT == "change-me-in-prod"


def test_session12_agent_recovery_defaults(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_SEARCH_DISTANCE_THRESHOLD", raising=False)
    monkeypatch.delenv("AGENT_RECOVERY_RELIABILITY_THRESHOLD", raising=False)
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="test",
        LLM_PROVIDER="openai",
    )
    assert settings.AGENT_SEARCH_DISTANCE_THRESHOLD == 0.45
    assert settings.AGENT_RECOVERY_RELIABILITY_THRESHOLD == 0.35
