"""Session 15 health endpoints: liveness stays cheap; readiness checks deps."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_health_ready_200_when_deps_ok(client: TestClient, monkeypatch) -> None:
    fake_conn = MagicMock()
    fake_engine = MagicMock()
    fake_engine.connect.return_value.__enter__.return_value = fake_conn
    fake_engine.connect.return_value.__exit__.return_value = False

    fake_redis = MagicMock()
    fake_redis.ping.return_value = True

    monkeypatch.setattr(
        "app.api.routers.health.create_engine_from_settings",
        lambda: fake_engine,
    )
    monkeypatch.setattr(
        "app.api.routers.health.redis.from_url",
        lambda *a, **k: fake_redis,
    )

    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"] == {"postgres": "ok", "redis": "ok"}


def test_health_ready_503_when_postgres_down(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.routers.health.create_engine_from_settings",
        lambda: (_ for _ in ()).throw(RuntimeError("db down")),
    )
    fake_redis = MagicMock()
    fake_redis.ping.return_value = True
    monkeypatch.setattr(
        "app.api.routers.health.redis.from_url",
        lambda *a, **k: fake_redis,
    )

    response = client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["postgres"].startswith("error:")
    assert body["checks"]["redis"] == "ok"
