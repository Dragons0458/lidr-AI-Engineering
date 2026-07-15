"""HTTP contract tests for POST /v1/estimate/agent/graph."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import app.api.routers.estimate_graph as router
import app.api.security as security
from app.api.rate_limiting import limiter
from app.domain.graph_estimation import (
    GraphEstimationError,
    GraphRuntimeUnavailableError,
)
from app.domain.schemas.graph_estimation import GraphEstimationResponse
from app.generation.agentic.agent_schemas import AgentEstimate
from app.main import app

KEY = "s13-estimate-key"
BODY = {"estimation_id": "est-1", "transcript": "Build a customer portal"}


@pytest.fixture(autouse=True)
def configured_router(monkeypatch):
    settings = SimpleNamespace(ESTIMATE_API_KEY=KEY, RETRIEVAL_API_KEY="other")
    monkeypatch.setattr(security, "get_settings", lambda: settings)
    limiter._storage.reset()
    # Avoid depending on a live Postgres checkpointer during unit tests.
    monkeypatch.setattr(router, "get_graph_runtime", lambda request: object())


@pytest.fixture
def client():
    return TestClient(app)


def _headers():
    return {"X-API-Key": KEY}


def _response() -> GraphEstimationResponse:
    return GraphEstimationResponse(
        estimate=AgentEstimate(
            components=[
                {
                    "name": "Auth",
                    "estimated_hours": 46.0,
                    "cited_chunk_ids": [1],
                    "rationale": "ok",
                }
            ],
            total_hours=46.0,
            assumptions=[],
            confidence="medium",
        ),
        status="validated",
    )


def test_graph_endpoint_requires_estimate_api_key(client):
    assert client.post("/v1/estimate/agent/graph", json=BODY).status_code == 401


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"estimation_id": "e1"},
        {"transcript": "hello"},
        {"estimation_id": "", "transcript": "hello"},
        {"estimation_id": "e1", "transcript": ""},
        {"estimation_id": "e1", "transcript": "hello", "status": "validated"},
    ],
)
def test_invalid_payloads_return_422(client, payload):
    assert (
        client.post(
            "/v1/estimate/agent/graph", json=payload, headers=_headers()
        ).status_code
        == 422
    )


def test_status_literal_only_allows_documented_values():
    from app.domain.schemas.graph_estimation import GraphEstimationResponse

    with pytest.raises(Exception):
        GraphEstimationResponse(
            estimate=AgentEstimate(
                components=[],
                total_hours=0,
                assumptions=[],
                confidence="low",
            ),
            status="approved",  # type: ignore[arg-type]
        )


def test_happy_path_returns_estimate_and_status(client, monkeypatch):
    monkeypatch.setattr(
        router,
        "run_graph_estimation",
        AsyncMock(return_value=_response()),
    )
    response = client.post("/v1/estimate/agent/graph", json=BODY, headers=_headers())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "validated"
    assert body["estimate"]["total_hours"] == 46.0


def test_runtime_unavailable_returns_503(client, monkeypatch):
    monkeypatch.setattr(
        router,
        "run_graph_estimation",
        AsyncMock(side_effect=GraphRuntimeUnavailableError("unavailable")),
    )
    response = client.post("/v1/estimate/agent/graph", json=BODY, headers=_headers())
    assert response.status_code == 503
    assert (
        "not available" in response.json()["detail"].lower()
        or "unavailable" in response.json()["detail"].lower()
    )


def test_graph_failure_returns_502(client, monkeypatch):
    monkeypatch.setattr(
        router,
        "run_graph_estimation",
        AsyncMock(side_effect=GraphEstimationError("boom")),
    )
    response = client.post("/v1/estimate/agent/graph", json=BODY, headers=_headers())
    assert response.status_code == 502


def test_openapi_includes_graph_path(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/estimate/agent/graph" in paths
