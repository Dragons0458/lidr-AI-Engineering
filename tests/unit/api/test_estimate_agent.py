"""HTTP contract tests for the Session 12 hybrid-agent endpoints."""

from types import SimpleNamespace
from unittest.mock import ANY

import pytest
from fastapi.testclient import TestClient

import app.api.routers.estimate_agent as router
import app.api.security as security
from app.api.rate_limiting import limiter
from app.domain.agent_estimation import OpenAIClientMissingError, RecoveryAgentError
from app.domain.schemas.agent_trace import AgentStep, AgentTrace
from app.generation.rag.schemas import (
    Estimate,
    GenerateResult,
    TaskHoursEstimate,
    TaskHoursResult,
)
from app.main import app

KEY = "s12-estimate-key"
QUERY = {"function": "Build a customer portal"}
STRUCTURE_BODY = {"query": QUERY}
HOURS_BODY = {"modules": [{"name": "Core", "tasks": [{"name": "Build"}]}]}


@pytest.fixture(autouse=True)
def configured_router(monkeypatch):
    settings = SimpleNamespace(
        ESTIMATE_API_KEY=KEY,
        RETRIEVAL_API_KEY="other",
        AGENT_MODEL="gpt-5",
        AGENT_REASONING_EFFORT="medium",
        AGENT_MAX_ITERATIONS=10,
        AGENT_RECOVERY_RELIABILITY_THRESHOLD=0.35,
    )
    runtime = SimpleNamespace(
        effective_task_hours_top_k=lambda: 5,
        effective_task_hours_distance_threshold=lambda: 0.45,
        effective_search_mode=lambda: "hybrid",
        effective_rerank=lambda: True,
    )
    monkeypatch.setattr(security, "get_settings", lambda: settings)
    monkeypatch.setattr(router, "get_settings", lambda: settings)
    monkeypatch.setattr(router, "get_runtime_retrieval_config", lambda: runtime)
    monkeypatch.setattr(router, "get_embedder", lambda: object())
    monkeypatch.setattr(router, "get_async_openai_client", lambda: object())
    limiter._storage.reset()


@pytest.fixture
def client():
    return TestClient(app)


def _headers():
    return {"X-API-Key": KEY}


def _structure_result():
    return GenerateResult(
        estimate=Estimate(
            confidence="high",
            reasoning="clear",
            modules=[{"name": "Core", "tasks": [{"name": "Build"}]}],
        ),
        coherent=True,
        agent_trace=AgentTrace(
            steps=[
                AgentStep(
                    step=1,
                    tool="propose_structure",
                    tool_args={},
                    observation="proposed 1 module",
                )
            ]
        ),
    )


def test_both_endpoints_require_estimate_api_key(client):
    assert (
        client.post("/v1/estimate/agent/structure", json=STRUCTURE_BODY).status_code
        == 401
    )
    assert client.post("/v1/estimate/agent/hours", json=HOURS_BODY).status_code == 401


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/estimate/agent/structure", {}),
        (
            "/v1/estimate/agent/structure",
            {"query": QUERY, "persona": "x" * 2001},
        ),
        ("/v1/estimate/agent/hours", {"modules": []}),
        ("/v1/estimate/agent/hours", {**HOURS_BODY, "max_iterations": 21}),
        ("/v1/estimate/agent/hours", {**HOURS_BODY, "search_top_k": 31}),
        (
            "/v1/estimate/agent/hours",
            {**HOURS_BODY, "search_distance_threshold": 2.1},
        ),
        ("/v1/estimate/agent/hours", {**HOURS_BODY, "unknown": True}),
    ],
)
def test_invalid_payloads_return_422(client, path, payload):
    assert client.post(path, json=payload, headers=_headers()).status_code == 422


def test_structure_ignores_hours_knobs_and_forwards_only_phase_fields(
    client, monkeypatch
):
    captured = {}

    async def fake(query, **kwargs):
        captured.update(kwargs)
        return _structure_result()

    monkeypatch.setattr(router, "agent_propose_structure", fake)
    payload = {
        "query": QUERY,
        "model": "gpt-5-mini",
        "reasoning_effort": "low",
        "persona": "Be concise",
        "max_iterations": 20,
        "search_top_k": 30,
        "search_distance_threshold": 2,
    }
    response = client.post(
        "/v1/estimate/agent/structure", json=payload, headers=_headers()
    )
    assert response.status_code == 200
    assert captured == {
        "client": ANY,
        "model": "gpt-5-mini",
        "reasoning_effort": "low",
        "persona": "Be concise",
    }
    body = response.json()
    assert body["estimate"]["modules"][0]["tasks"][0]["engineer_days"] is None
    assert body["agent_trace"]["steps"][0]["tool"] == "propose_structure"


def test_hours_forwards_overrides(client, monkeypatch):
    captured = {}

    async def fake(modules, **kwargs):
        captured.update(kwargs)
        return TaskHoursResult(tasks=[], agent_trace=AgentTrace())

    monkeypatch.setattr(router, "agent_estimate_task_hours", fake)
    payload = {
        **HOURS_BODY,
        "model": "gpt-5-mini",
        "reasoning_effort": "high",
        "max_iterations": 4,
        "search_top_k": 8,
        "search_distance_threshold": 0.3,
        "persona": "Skeptical",
    }
    assert (
        client.post(
            "/v1/estimate/agent/hours", json=payload, headers=_headers()
        ).status_code
        == 200
    )
    assert captured["model"] == "gpt-5-mini"
    assert captured["reasoning_effort"] == "high"
    assert captured["max_iterations"] == 4
    assert captured["top_k"] == 8
    assert captured["distance_threshold"] == 0.3
    assert captured["persona"] == "Skeptical"


def test_hours_uses_runtime_defaults_and_allows_no_client_without_flags(
    client, monkeypatch
):
    captured = {}
    monkeypatch.setattr(router, "get_async_openai_client", lambda: None)

    async def fake(modules, **kwargs):
        captured.update(kwargs)
        return TaskHoursResult(
            tasks=[
                TaskHoursEstimate(
                    module="Core", task="Build", estimated_hours=20, has_match=True
                )
            ],
            agent_trace=AgentTrace(),
        )

    monkeypatch.setattr(router, "agent_estimate_task_hours", fake)
    response = client.post(
        "/v1/estimate/agent/hours", json=HOURS_BODY, headers=_headers()
    )
    assert response.status_code == 200
    assert captured["client"] is None
    assert captured["top_k"] == 5
    assert captured["distance_threshold"] == 0.45
    assert captured["search_mode"] == "hybrid"
    assert captured["rerank"] is True


def test_hours_maps_missing_client_to_500(client, monkeypatch):
    async def fail(*args, **kwargs):
        raise OpenAIClientMissingError("client required")

    monkeypatch.setattr(router, "agent_estimate_task_hours", fail)
    assert (
        client.post(
            "/v1/estimate/agent/hours", json=HOURS_BODY, headers=_headers()
        ).status_code
        == 500
    )


def test_hours_without_embedder_returns_500(client, monkeypatch):
    monkeypatch.setattr(router, "get_embedder", lambda: None)
    assert (
        client.post(
            "/v1/estimate/agent/hours", json=HOURS_BODY, headers=_headers()
        ).status_code
        == 500
    )


@pytest.mark.parametrize(
    ("path", "target", "error"),
    [
        (
            "/v1/estimate/agent/structure",
            "agent_propose_structure",
            RuntimeError("model"),
        ),
        (
            "/v1/estimate/agent/hours",
            "agent_estimate_task_hours",
            RecoveryAgentError("loop"),
        ),
    ],
)
def test_model_and_loop_failures_return_502(client, monkeypatch, path, target, error):
    async def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(router, target, fail)
    payload = STRUCTURE_BODY if path.endswith("structure") else HOURS_BODY
    assert client.post(path, json=payload, headers=_headers()).status_code == 502


def test_rate_limit_remains_429(client, monkeypatch):
    async def fake(*args, **kwargs):
        return _structure_result()

    monkeypatch.setattr(router, "agent_propose_structure", fake)
    responses = [
        client.post(
            "/v1/estimate/agent/structure", json=STRUCTURE_BODY, headers=_headers()
        )
        for _ in range(16)
    ]
    assert responses[-1].status_code == 429
    assert responses[-1].headers["Retry-After"] == "60"


def test_agent_paths_are_present_in_openapi(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/estimate/agent/structure" in paths
    assert "/v1/estimate/agent/hours" in paths
