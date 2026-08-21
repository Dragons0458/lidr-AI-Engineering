"""HTTP contract for GET /v1/observability/* — fake repository, no Postgres, no LLM."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.api.security as security
from app.api.rate_limiting import limiter
from app.api.routers.observability import get_metrics_repo
from app.foundation.observability.metrics import RequestMetrics
from app.main import app

KEY = "s16-estimate-key"


class EmptyRepo:
    def aggregate_metrics(self, **kwargs):
        return {
            "window_hours": kwargs.get("window_hours", 24),
            "requests": 0,
            "error_rate": 0.0,
            "latency_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "n": 0},
            "cost_usd": {"total": 0.0, "mean_per_request": 0.0, "p95": 0.0},
            "tokens": {"prompt": 0, "completion": 0},
            "abstention_rate": 0.0,
            "cache_hit_rate": 0.0,
            "by_route": [],
            "series": [],
        }

    def list_metrics(self, **kwargs):
        return []


class PopulatedRepo(EmptyRepo):
    def aggregate_metrics(self, **kwargs):
        payload = super().aggregate_metrics(**kwargs)
        payload.update(
            {
                "requests": 2,
                "error_rate": 0.0,
                "latency_ms": {"mean": 100.0, "p50": 90.0, "p95": 120.0, "n": 2},
                "cost_usd": {"total": 0.04, "mean_per_request": 0.02, "p95": 0.03},
                "tokens": {"prompt": 10, "completion": 4},
            }
        )
        return payload

    def list_metrics(self, **kwargs):
        prefix = kwargs.get("request_id_prefix")
        row = RequestMetrics(
            request_id="s16-abc-S16-01-1",
            route="/v1/estimate/from-transcript",
            http_status=200,
            status="ok",
            latency_ms=100.0,
            llm_calls=3,
            estimated_cost_usd=0.02,
        )
        if prefix and not row.request_id.startswith(prefix):
            return []
        return [row]


@pytest.fixture(autouse=True)
def configured_router(monkeypatch):
    settings = SimpleNamespace(
        ESTIMATE_API_KEY=KEY,
        RETRIEVAL_API_KEY="other",
        AI_SERVICE_TOKEN=KEY,
        effective_service_token=KEY,
    )
    monkeypatch.setattr(security, "get_settings", lambda: settings)
    limiter._storage.reset()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _headers():
    return {"X-API-Key": KEY}


def test_metrics_requires_api_key(client):
    assert client.get("/v1/observability/metrics").status_code == 401


def test_empty_window_returns_zeros_not_500(client):
    app.dependency_overrides[get_metrics_repo] = lambda: EmptyRepo()
    response = client.get("/v1/observability/metrics", headers=_headers())
    assert response.status_code == 200
    body = response.json()
    assert body["requests"] == 0
    assert body["error_rate"] == 0.0
    assert body["latency_ms"]["n"] == 0
    assert body["series"] == []


def test_metrics_shape_and_filters(client):
    app.dependency_overrides[get_metrics_repo] = lambda: PopulatedRepo()
    response = client.get(
        "/v1/observability/metrics",
        headers=_headers(),
        params={"window_hours": 1, "route": "/v1/estimate/from-transcript"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["window_hours"] == 1
    assert body["requests"] == 2
    assert "latency_ms" in body and "p95" in body["latency_ms"]
    assert "cost_usd" in body and "mean_per_request" in body["cost_usd"]


def test_requests_list_and_prefix_filter(client):
    app.dependency_overrides[get_metrics_repo] = lambda: PopulatedRepo()
    ok = client.get("/v1/observability/requests", headers=_headers())
    assert ok.status_code == 200
    assert ok.json()[0]["request_id"] == "s16-abc-S16-01-1"
    assert "transcript" not in ok.json()[0]

    filtered = client.get(
        "/v1/observability/requests",
        headers=_headers(),
        params={"request_id_prefix": "s16-abc-S16-01"},
    )
    assert filtered.status_code == 200
    assert len(filtered.json()) == 1
