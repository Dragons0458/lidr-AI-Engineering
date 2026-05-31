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
