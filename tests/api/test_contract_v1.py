"""Session 15 — the web→AI contract as a verifiable artefact (no network, no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api.security as security
from app.dependencies import get_semantic_retriever
from app.main import app

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs" / "contract" / "web-consumed-routes.json"

EST_KEY = "ci-estimate-key"
RET_KEY = "ci-retrieval-key"


def _load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_the_contract_artifact_is_present_and_non_empty() -> None:
    assert CONTRACT_PATH.is_file()
    contract = _load_contract()
    assert len(contract["routes"]) >= 25
    assert len(contract["probes"]) >= 2


def test_every_route_the_web_ui_calls_exists() -> None:
    spec_paths = app.openapi()["paths"]
    missing: list[str] = []
    for route in _load_contract()["routes"]:
        path = route["path"]
        method = route["method"].lower()
        if path not in spec_paths or method not in spec_paths[path]:
            missing.append(f"{method.upper()} {path}")
    assert missing == []


@pytest.mark.require_service_token
@pytest.mark.parametrize("path", ["/health", "/health/ready"])
def test_probes_never_require_the_service_token(client: TestClient, path: str) -> None:
    assert client.get(path).status_code != 401


@pytest.mark.require_service_token
def test_401_when_the_service_token_is_missing(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "RETRIEVAL_API_KEY": RET_KEY,
                "ESTIMATE_API_KEY": EST_KEY,
                "AI_SERVICE_TOKEN": EST_KEY,
                "effective_service_token": EST_KEY,
            },
        )(),
    )
    response = client.post(
        "/api/v1/estimate",
        json={
            "description": "A meeting summary long enough to pass validation.",
            "project_type": "web_saas",
            "detail_level": "medium",
            "output_format": "line_items",
        },
    )
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "X-API-Key"


@pytest.mark.require_service_token
def test_401_on_retrieval_with_the_wrong_key(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "RETRIEVAL_API_KEY": RET_KEY,
                "ESTIMATE_API_KEY": EST_KEY,
                "AI_SERVICE_TOKEN": EST_KEY,
                "effective_service_token": EST_KEY,
            },
        )(),
    )
    response = client.post(
        "/v1/retrieval/search",
        json={"query_text": "ecommerce storefront with card checkout"},
        headers={"X-API-Key": EST_KEY},
    )
    assert response.status_code == 401


def test_422_on_invalid_input(client: TestClient) -> None:
    response = client.post(
        "/api/v1/estimate",
        json={
            "description": "short",
            "project_type": "web_saas",
            "detail_level": "medium",
            "output_format": "line_items",
        },
    )
    assert response.status_code == 422


def test_503_when_a_dependency_is_unavailable() -> None:
    app.dependency_overrides[get_semantic_retriever] = lambda: None
    try:
        response = TestClient(app).post("/search", json={"query": "anything", "k": 5})
    finally:
        app.dependency_overrides.pop(get_semantic_retriever, None)
    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding service is not available."


def test_500_is_still_reserved_for_genuine_failures() -> None:
    class ExplodingRetriever:
        async def search(self, **kwargs):
            raise RuntimeError("embeddings API down")

    app.dependency_overrides[get_semantic_retriever] = lambda: ExplodingRetriever()
    try:
        response = TestClient(app).post("/search", json={"query": "anything", "k": 5})
    finally:
        app.dependency_overrides.pop(get_semantic_retriever, None)
    assert response.status_code == 500
