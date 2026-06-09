import sys
from pathlib import Path

import fakeredis
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import dependencies  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.foundation.llm.runtime_config import RuntimeModelConfig  # noqa: E402
from app.foundation.llm.wrapper import LLMWrapper  # noqa: E402
from app.generation.cag.exact import EstimationCache  # noqa: E402
from app.main import app  # noqa: E402


def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def client() -> TestClient:
    """Provide a FastAPI test client configured with the application."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def disable_session5_env_features(monkeypatch) -> None:
    """Keep unit tests deterministic regardless of developer .env Session 5 flags."""
    for module_path in (
        "app.api.sessions",
        "app.domain.estimation_service",
        "app.api.estimations",
    ):
        monkeypatch.setattr(
            f"{module_path}.settings.TIER_RESOLUTION_ENABLED",
            False,
            raising=False,
        )
        monkeypatch.setattr(
            f"{module_path}.settings.MEMORY_COMPRESSION_ENABLED",
            False,
            raising=False,
        )


@pytest.fixture(autouse=True)
def isolated_llm_wrapper(monkeypatch) -> None:
    """In-memory Redis and a wrapper with cache disabled for deterministic unit tests."""
    from app.dependencies import (
        get_cache,
        get_catalog,
        get_filesystem_loader,
        get_llm_wrapper,
        get_parser_registry,
        get_runtime_config,
        get_semantic_cache,
    )

    redis_client = fakeredis.FakeRedis(decode_responses=True)
    cache = EstimationCache(redis_client, ttl=60)
    settings = get_settings()
    runtime_config = RuntimeModelConfig(redis_client, settings)
    wrapper = LLMWrapper(
        primary_model=settings.PRIMARY_MODEL,
        fallback_model=None,
        timeout=settings.LLM_TIMEOUT,
        num_retries=settings.LLM_RETRIES,
        cache=cache,
        cache_enabled=False,
        runtime_config=runtime_config,
    )

    get_cache.cache_clear()
    get_llm_wrapper.cache_clear()
    get_runtime_config.cache_clear()
    get_semantic_cache.cache_clear()
    get_catalog.cache_clear()
    get_filesystem_loader.cache_clear()
    get_parser_registry.cache_clear()
    monkeypatch.setattr(dependencies, "get_cache", lambda: cache)
    monkeypatch.setattr(dependencies, "get_llm_wrapper", lambda: wrapper)
    monkeypatch.setattr(dependencies, "get_runtime_config", lambda: runtime_config)
    monkeypatch.setattr(dependencies, "get_semantic_cache", lambda: None)
    # estimation_service imports get_llm_wrapper by name; patch that binding too.
    monkeypatch.setattr(
        "app.domain.estimation_service.get_llm_wrapper", lambda: wrapper
    )
    monkeypatch.setattr(
        "app.domain.estimation_service.get_semantic_cache", lambda: None
    )
    monkeypatch.setattr("app.domain.estimation_service.get_cache", lambda: cache)
    monkeypatch.setattr(
        "app.domain.estimation_service.settings.INPUT_GUARDRAILS_ENABLED",
        False,
    )
    yield
    get_cache.cache_clear()
    get_llm_wrapper.cache_clear()
    get_runtime_config.cache_clear()
    get_semantic_cache.cache_clear()
    get_catalog.cache_clear()
    get_filesystem_loader.cache_clear()
    get_parser_registry.cache_clear()
