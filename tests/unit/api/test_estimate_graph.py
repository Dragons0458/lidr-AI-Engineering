"""HTTP contract tests for the Session 13 multi-agent graph endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import app.api.routers.estimate_graph as router
import app.api.security as security
from app.api.rate_limiting import limiter
from app.domain.graph_estimation import (
    GraphConflictError,
    GraphEstimationError,
    GraphNotFoundError,
    GraphRuntimeUnavailableError,
)
from app.domain.schemas.graph_estimation import (
    GraphProgress,
    GraphRunState,
    PendingGate,
)
from app.generation.agentic.graph.schemas import CommercialProposal
from app.main import app

KEY = "s13-estimate-key"
BODY = {"estimation_id": "est-1", "transcript": "Build a customer portal"}


@pytest.fixture(autouse=True)
def configured_router(monkeypatch):
    settings = SimpleNamespace(ESTIMATE_API_KEY=KEY, RETRIEVAL_API_KEY="other")
    monkeypatch.setattr(security, "get_settings", lambda: settings)
    limiter._storage.reset()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def _headers():
    return {"X-API-Key": KEY}


def _run_state(*, paused: bool = True) -> GraphRunState:
    return GraphRunState(
        estimation_id="est-1",
        state="paused" if paused else "completed",
        pending_gate=PendingGate(
            gate="structure_review",
            estimation_id="est-1",
            payload={"structure": {"modules": []}},
        )
        if paused
        else None,
        status="validated" if not paused else None,
    )


def _wire_runtime(client_app, runtime=object(), deps=object()):
    client_app.state.graph_runtime = runtime
    client_app.state.multiagent_deps = deps


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
    ],
)
def test_invalid_payloads_return_422(client, payload):
    assert (
        client.post(
            "/v1/estimate/agent/graph", json=payload, headers=_headers()
        ).status_code
        == 422
    )


def test_start_returns_run_state(client, monkeypatch):
    _wire_runtime(client.app)
    monkeypatch.setattr(
        router,
        "start_graph_run",
        AsyncMock(return_value=_run_state(paused=True)),
    )
    response = client.post("/v1/estimate/agent/graph", json=BODY, headers=_headers())
    assert response.status_code == 200
    assert response.json()["state"] == "paused"


def test_resume_conflict_returns_409(client, monkeypatch):
    _wire_runtime(client.app)
    monkeypatch.setattr(
        router,
        "resume_graph_run",
        AsyncMock(side_effect=GraphConflictError("already completed")),
    )
    response = client.post(
        "/v1/estimate/agent/graph/est-1/resume",
        json={"decision": {"approved": True}},
        headers=_headers(),
    )
    assert response.status_code == 409


def test_state_unknown_returns_404(client, monkeypatch):
    _wire_runtime(client.app)
    monkeypatch.setattr(
        router,
        "read_graph_state",
        AsyncMock(side_effect=GraphNotFoundError("unknown")),
    )
    response = client.get(
        "/v1/estimate/agent/graph/est-1/state",
        headers=_headers(),
    )
    assert response.status_code == 404


def test_runtime_unavailable_returns_503(client, monkeypatch):
    monkeypatch.setattr(
        router,
        "start_graph_run",
        AsyncMock(side_effect=GraphRuntimeUnavailableError("down")),
    )
    response = client.post("/v1/estimate/agent/graph", json=BODY, headers=_headers())
    assert response.status_code == 503


def test_graph_failure_returns_502(client, monkeypatch):
    _wire_runtime(client.app)
    monkeypatch.setattr(
        router,
        "start_graph_run",
        AsyncMock(side_effect=GraphEstimationError("boom")),
    )
    response = client.post("/v1/estimate/agent/graph", json=BODY, headers=_headers())
    assert response.status_code == 502


def test_stream_start_returns_202_and_resets_activity(client, monkeypatch):
    runtime = SimpleNamespace(graph=MagicMock())
    activity = MagicMock()
    _wire_runtime(client.app, runtime=runtime)
    monkeypatch.setattr(
        "app.dependencies.get_graph_activity",
        lambda: activity,
    )
    client.app.dependency_overrides[router.get_graph_activity] = lambda: activity
    response = client.post(
        "/v1/estimate/agent/graph/stream",
        json=BODY,
        headers=_headers(),
    )
    assert response.status_code == 202
    assert response.json()["state"] == "running"
    activity.reset.assert_called_once_with("est-1")


def test_progress_returns_graph_progress(client, monkeypatch):
    _wire_runtime(client.app)
    monkeypatch.setattr(
        router,
        "read_graph_progress",
        AsyncMock(
            return_value=GraphProgress(
                estimation_id="est-1",
                state="paused",
                activity=[],
            )
        ),
    )
    response = client.get(
        "/v1/estimate/agent/graph/est-1/progress",
        headers=_headers(),
    )
    assert response.status_code == 200
    assert response.json()["state"] == "paused"


def test_proposal_returns_response(client, monkeypatch):
    deps = SimpleNamespace(
        propose=AsyncMock(
            return_value=CommercialProposal(
                title="T",
                executive_summary="S",
                scope=["A"],
                total_engineer_days=10,
                body_markdown="# Hi",
            )
        )
    )
    _wire_runtime(client.app, runtime=object(), deps=deps)
    monkeypatch.setattr(
        router,
        "draft_graph_proposal",
        AsyncMock(
            return_value=CommercialProposal(
                title="T",
                executive_summary="S",
                scope=["A"],
                total_engineer_days=10,
                body_markdown="# Hi",
            )
        ),
    )
    response = client.post(
        "/v1/estimate/agent/graph/est-1/proposal",
        headers=_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "T"
    assert body["body_markdown"] == "# Hi"


def test_proposal_conflict_returns_409(client, monkeypatch):
    deps = SimpleNamespace(propose=AsyncMock())
    _wire_runtime(client.app, runtime=object(), deps=deps)
    monkeypatch.setattr(
        router,
        "draft_graph_proposal",
        AsyncMock(side_effect=GraphConflictError("no estimate")),
    )
    response = client.post(
        "/v1/estimate/agent/graph/est-1/proposal",
        headers=_headers(),
    )
    assert response.status_code == 409
