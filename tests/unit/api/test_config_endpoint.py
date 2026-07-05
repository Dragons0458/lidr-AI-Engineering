from __future__ import annotations

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.dependencies import get_runtime_config
from app.foundation.llm.runtime_config import RuntimeModelConfig
from app.main import app


def make_settings(**overrides) -> Settings:
    defaults = {
        "OPENAI_API_KEY": "sk-test",
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "PRIMARY_MODEL": "gpt-4o-mini",
        "LLM_PROVIDER": "openai",
    }
    return Settings(_env_file=None, **{**defaults, **overrides})


@pytest.fixture
def fake_store():
    settings = make_settings()
    store = RuntimeModelConfig(fakeredis.FakeRedis(decode_responses=True), settings)
    app.dependency_overrides[get_runtime_config] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    yield store
    app.dependency_overrides.clear()


@pytest.fixture
def client(fake_store) -> TestClient:
    return TestClient(app)


def test_get_returns_full_snapshot(client) -> None:
    response = client.get("/api/v1/config/models")
    assert response.status_code == 200
    body = response.json()
    assert body["models"]["PRIMARY_MODEL"]["effective"] == "gpt-4o-mini"
    assert set(body["models"]) == {
        "PRIMARY_MODEL",
        "FALLBACK_MODEL",
        "CRITIC_MODEL",
        "COMPRESSION_MODEL",
        "PROPOSITIONAL_CHUNKER_MODEL",
        "CONTEXTUAL_CHUNKER_MODEL",
        "HALLUCINATION_JUDGE_MODEL",
        "AUGMENTATION_MODEL",
    }
    assert "gpt-4o" in body["available_models"]


def test_put_overrides_and_returns_fresh_snapshot(client, fake_store) -> None:
    response = client.put(
        "/api/v1/config/models", json={"models": {"PRIMARY_MODEL": "gpt-4o"}}
    )
    assert response.status_code == 200
    assert response.json()["models"]["PRIMARY_MODEL"]["overridden"] is True
    assert fake_store.effective("PRIMARY_MODEL") == "gpt-4o"


def test_put_unknown_key_is_422(client) -> None:
    response = client.put(
        "/api/v1/config/models", json={"models": {"EMBEDDING_MODEL": "gpt-4o"}}
    )
    assert response.status_code == 422


def test_put_model_with_missing_provider_key_is_400(fake_store) -> None:
    settings = make_settings(ANTHROPIC_API_KEY=None)
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)
    response = client.put(
        "/api/v1/config/models",
        json={"models": {"PRIMARY_MODEL": "claude-sonnet-4-5"}},
    )
    assert response.status_code == 400
