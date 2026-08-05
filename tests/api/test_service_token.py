"""Session 15 — global service-token gate + public-path allowlist."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.api.security as security
from app.api.security import is_public_path
from app.main import app

SERVICE_KEY = "service-secret"
RETRIEVAL_KEY = "retrieval-secret"


@pytest.fixture(autouse=True)
def stub_settings(monkeypatch):
    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "RETRIEVAL_API_KEY": RETRIEVAL_KEY,
                "ESTIMATE_API_KEY": SERVICE_KEY,
                "AI_SERVICE_TOKEN": SERVICE_KEY,
                "effective_service_token": SERVICE_KEY,
            },
        )(),
    )
    yield


@pytest.fixture
def client():
    # Keep the global gate active for this module.
    return TestClient(app)


@pytest.mark.require_service_token
@pytest.mark.parametrize(
    "path",
    ["/health", "/health/ready", "/docs", "/openapi.json", "/redoc"],
)
def test_public_paths_need_no_token(client, path):
    assert client.get(path).status_code != 401


@pytest.mark.require_service_token
def test_protected_route_without_token_is_401(client):
    response = client.get("/api/v1/config/models")
    assert response.status_code == 401


@pytest.mark.require_service_token
def test_protected_route_accepts_service_token(client):
    from app.dependencies import get_runtime_config
    import fakeredis
    from app.config import Settings
    from app.foundation.llm.runtime_config import RuntimeModelConfig

    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="sk-test",
        PRIMARY_MODEL="gpt-4o-mini",
        ESTIMATE_API_KEY=SERVICE_KEY,
        AI_SERVICE_TOKEN=SERVICE_KEY,
    )
    store = RuntimeModelConfig(fakeredis.FakeRedis(decode_responses=True), settings)
    app.dependency_overrides[get_runtime_config] = lambda: store
    try:
        response = client.get(
            "/api/v1/config/models",
            headers={"X-API-Key": SERVICE_KEY},
        )
        assert response.status_code == 200
    finally:
        app.dependency_overrides.pop(get_runtime_config, None)


@pytest.mark.require_service_token
def test_allowlist_rejects_unknown_public_paths():
    """Any new public path must be added to PUBLIC_PATH_PREFIXES deliberately."""
    assert is_public_path("/health")
    assert is_public_path("/health/ready")
    assert is_public_path("/docs")
    assert is_public_path("/openapi.json")
    assert not is_public_path("/api/v1/estimate")
    assert not is_public_path("/v1/estimate/from-transcript")
    assert not is_public_path("/embeddings/ingest")
