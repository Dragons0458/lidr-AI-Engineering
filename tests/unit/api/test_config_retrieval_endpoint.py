"""Tests for GET/PUT /api/v1/config/retrieval."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_runtime_retrieval_config
from app.foundation.llm.runtime_config import (
    RuntimeConfigUnavailable,
    RuntimeRetrievalConfig,
)
from app.main import app


@pytest.fixture
def client(monkeypatch) -> TestClient:
    import fakeredis

    from tests.unit.foundation.test_runtime_retrieval_config import make_settings

    store = RuntimeRetrievalConfig(
        fakeredis.FakeRedis(decode_responses=True), make_settings()
    )
    app.dependency_overrides[get_runtime_retrieval_config] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_get_retrieval_returns_snapshot(client: TestClient) -> None:
    response = client.get("/api/v1/config/retrieval")
    assert response.status_code == 200
    body = response.json()
    assert "retrieval" in body
    assert "RETRIEVAL_SEARCH_MODE" in body["retrieval"]


def test_put_retrieval_updates_search_mode(client: TestClient) -> None:
    response = client.put(
        "/api/v1/config/retrieval",
        json={"search_mode": "hybrid"},
    )
    assert response.status_code == 200
    assert (
        response.json()["retrieval"]["RETRIEVAL_SEARCH_MODE"]["effective"] == "hybrid"
    )


def test_put_retrieval_rejects_invalid_search_mode(client: TestClient) -> None:
    response = client.put(
        "/api/v1/config/retrieval",
        json={"search_mode": "invalid"},
    )
    assert response.status_code == 422


def test_put_retrieval_s11_toggles(client: TestClient) -> None:
    response = client.put(
        "/api/v1/config/retrieval",
        json={
            "hallucination_gate_enabled": False,
            "augmentation_enabled": False,
            "synthesis_enabled": False,
        },
    )
    assert response.status_code == 200
    body = response.json()["retrieval"]
    assert body["HALLUCINATION_GATE_ENABLED"]["effective"] is False
    assert body["AUGMENTATION_ENABLED"]["effective"] is False
    assert body["SYNTHESIS_ENABLED"]["effective"] is False


def test_put_retrieval_503_when_redis_unavailable(monkeypatch) -> None:
    broken = MagicMock()
    broken.hget.return_value = None
    broken.hgetall.return_value = {}
    broken.hset.side_effect = RuntimeConfigUnavailable("down")

    from tests.unit.foundation.test_runtime_retrieval_config import make_settings

    store = RuntimeRetrievalConfig(broken, make_settings())
    app.dependency_overrides[get_runtime_retrieval_config] = lambda: store
    client = TestClient(app)
    try:
        response = client.put(
            "/api/v1/config/retrieval",
            json={"rerank": True},
        )
        assert response.status_code == 503
    finally:
        app.dependency_overrides.clear()
