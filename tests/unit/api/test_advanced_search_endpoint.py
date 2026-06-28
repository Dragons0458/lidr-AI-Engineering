"""Tests for POST /v1/retrieval/advanced-search."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.api.security as security
from app.generation.rag.retrieval.advanced_pipeline import AdvancedRetrievalOutcome
from app.generation.rag.retrieval.collections import Collection
from app.generation.rag.retrieval.query_transform import SubQuery
from app.generation.rag.retrieval.router import RoutingDecision
from app.generation.rag.schemas import RetrievedChunk
from app.main import app

RET_KEY = "retrieval-secret"


@pytest.fixture(autouse=True)
def stub_key(monkeypatch):
    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: type(
            "S", (), {"RETRIEVAL_API_KEY": RET_KEY, "ESTIMATE_API_KEY": "x"}
        )(),
    )


@pytest.fixture
def client():
    return TestClient(app)


def _headers():
    return {"X-API-Key": RET_KEY}


def test_advanced_search_requires_api_key(client):
    response = client.post(
        "/v1/retrieval/advanced-search",
        json={"query_text": "budget hours for OAuth authentication"},
    )
    assert response.status_code == 401


def test_advanced_search_validates_query_length(client):
    response = client.post(
        "/v1/retrieval/advanced-search",
        json={"query_text": "short"},
        headers=_headers(),
    )
    assert response.status_code == 422


def test_advanced_search_returns_shape(client, monkeypatch):
    import app.api.routers.retrieval_advanced as router

    chunk = RetrievedChunk(
        id=1,
        content="OAuth component",
        chunk_type="budget_component",
        distance=0.2,
        collection="budget",
    )
    outcome = AdvancedRetrievalOutcome(
        chunks=[chunk],
        routing=RoutingDecision(
            targets=[Collection.BUDGET],
            level="rules",
            reason="budget vocabulary",
        ),
        technique="direct",
        subqueries=[SubQuery(topic="auth", query="OAuth authentication budget")],
        cardinality={"budget": 1},
        low_confidence=False,
    )

    async def fake_advanced_retrieve(**kwargs):
        return outcome

    monkeypatch.setattr(router, "advanced_retrieve", fake_advanced_retrieve)
    monkeypatch.setattr(
        router,
        "get_embedder",
        lambda: type("E", (), {"embed_one": staticmethod(lambda t: [0.0] * 1536)})(),
    )

    response = client.post(
        "/v1/retrieval/advanced-search",
        json={"query_text": "budget hours for OAuth authentication fintech"},
        headers=_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["chunks"]) == 1
    assert body["technique"] == "direct"
    assert body["routing"]["level"] == "rules"
