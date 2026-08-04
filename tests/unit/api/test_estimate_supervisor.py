"""HTTP contract tests for Session 14 supervisor endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import app.api.routers.estimate_supervisor as router
import app.api.security as security
from app.api.rate_limiting import limiter
from app.domain.schemas.supervisor_estimation import (
    PendingHumanReview,
    SupervisorRunState,
)
from app.domain.supervisor_estimation import (
    SupervisorConflictError,
    SupervisorEstimationError,
    SupervisorNotFoundError,
    SupervisorRuntimeUnavailableError,
)
from app.main import app

KEY = "s14-estimate-key"
BODY = {
    "estimation_id": "est-1",
    "transcript": "We need a supplier portal with invoices and SAP sync. " * 5,
}


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


def _run_state(*, paused: bool = True) -> SupervisorRunState:
    return SupervisorRunState(
        estimation_id="est-1",
        state="paused" if paused else "completed",
        status="awaiting_human_review" if paused else "validated",
        pending_review=PendingHumanReview(
            estimation_id="est-1",
            reasons=["low_confidence"],
            confidence=0.3,
            threshold=0.6,
        )
        if paused
        else None,
    )


def test_supervisor_requires_estimate_api_key(client):
    assert client.post("/v1/estimate/agent/supervisor", json=BODY).status_code == 401


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"transcript": "short"},
        {"estimation_id": "e1"},
    ],
)
def test_invalid_payloads_return_422(client, payload):
    assert (
        client.post(
            "/v1/estimate/agent/supervisor", json=payload, headers=_headers()
        ).status_code
        == 422
    )


def test_start_returns_run_state(client, monkeypatch):
    monkeypatch.setattr(
        router,
        "start_supervisor_run",
        AsyncMock(return_value=_run_state(paused=True)),
    )
    response = client.post(
        "/v1/estimate/agent/supervisor", json=BODY, headers=_headers()
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "awaiting_human_review"
    assert body["state"] == "paused"


def test_start_503_when_runtime_missing(client, monkeypatch):
    monkeypatch.setattr(
        router,
        "start_supervisor_run",
        AsyncMock(side_effect=SupervisorRuntimeUnavailableError("no graph")),
    )
    response = client.post(
        "/v1/estimate/agent/supervisor", json=BODY, headers=_headers()
    )
    assert response.status_code == 503


def test_start_502_on_estimation_error(client, monkeypatch):
    monkeypatch.setattr(
        router,
        "start_supervisor_run",
        AsyncMock(side_effect=SupervisorEstimationError("boom")),
    )
    response = client.post(
        "/v1/estimate/agent/supervisor", json=BODY, headers=_headers()
    )
    assert response.status_code == 502


def test_resume_409_when_nothing_pending(client, monkeypatch):
    monkeypatch.setattr(
        router,
        "resume_supervisor_run",
        AsyncMock(side_effect=SupervisorConflictError("no pending")),
    )
    response = client.post(
        "/v1/estimate/agent/supervisor/est-1/resume",
        json={"decision": "approve"},
        headers=_headers(),
    )
    assert response.status_code == 409


def test_resume_typed_decision_rejects_typo(client):
    response = client.post(
        "/v1/estimate/agent/supervisor/est-1/resume",
        json={"decision": "aproove"},
        headers=_headers(),
    )
    assert response.status_code == 422


def test_state_404(client, monkeypatch):
    monkeypatch.setattr(
        router,
        "read_supervisor_state",
        AsyncMock(side_effect=SupervisorNotFoundError("missing")),
    )
    response = client.get(
        "/v1/estimate/agent/supervisor/missing/state", headers=_headers()
    )
    assert response.status_code == 404


def test_state_ok(client, monkeypatch):
    monkeypatch.setattr(
        router,
        "read_supervisor_state",
        AsyncMock(return_value=_run_state(paused=False)),
    )
    response = client.get(
        "/v1/estimate/agent/supervisor/est-1/state", headers=_headers()
    )
    assert response.status_code == 200
    assert response.json()["status"] == "validated"
