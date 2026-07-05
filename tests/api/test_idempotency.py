"""Idempotency tests for ``POST /v1/estimate/from-transcript`` (Session 9)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.api.security as security
import app.dependencies as deps
from app.generation.rag import estimator as orch
from app.generation.rag.idempotency import IdempotencyStore
from app.generation.rag.schemas import (
    Estimate,
    EstimationQuery,
    RetrievalResult,
    RetrievedChunk,
    SourceCitation,
)
from app.main import app

_KEY = "idem-estimate-key"


class CharEncoder:
    def encode(self, text: str) -> list[str]:
        return list(text)


@pytest.fixture
def generate_calls(monkeypatch):
    calls = {"generate": 0}
    store = IdempotencyStore(redis_client=None, ttl=3600)

    settings = type(
        "S",
        (),
        {
            "REFORMULATION_MODEL": "gpt-5-mini",
            "GENERATION_MODEL": "gpt-5",
            "GENERATION_REASONING_EFFORT": "medium",
            "RETRIEVAL_TOP_K": 10,
            "RETRIEVAL_DISTANCE_THRESHOLD": 0.6,
            "RETRIEVAL_RECALL_TOP_K": 50,
            "RERANK_TOP_N": 5,
            "RRF_K": 60,
            "MAX_CONTEXT_TOKENS": 100_000,
        },
    )()

    runtime = type(
        "R",
        (),
        {
            "effective_search_mode": staticmethod(lambda: "vector"),
            "effective_rerank": staticmethod(lambda: False),
            "effective_augmentation": staticmethod(lambda: False),
            "effective_hallucination_gate": staticmethod(lambda: False),
        },
    )()

    async def fake_reformulate(transcript):
        return EstimationQuery(function="ecommerce storefront", sector="ecommerce")

    async def fake_retrieve(**kwargs):
        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    id=1,
                    content="Checkout 140h",
                    sector="ecommerce",
                    project_year=2024,
                    chunk_type="budget_component",
                    distance=0.4,
                )
            ],
            low_confidence=False,
            candidates_evaluated=5,
        )

    async def fake_generate(context_block, structured_query, *, include_hours=True):
        calls["generate"] += 1
        return Estimate(
            total_engineer_days=18,
            duration_weeks=4,
            sources=[
                SourceCitation(source_id=1, relevance="primary", used_for="checkout")
            ],
            confidence="high",
            reasoning="grounded",
        )

    monkeypatch.setattr(orch, "get_settings", lambda: settings)
    monkeypatch.setattr(orch, "reformulate_query", fake_reformulate)
    monkeypatch.setattr(orch, "retrieve", fake_retrieve)
    monkeypatch.setattr(orch, "generate_estimate", fake_generate)
    monkeypatch.setattr(deps, "get_runtime_retrieval_config", lambda: runtime)
    monkeypatch.setattr(
        deps,
        "get_embedder",
        lambda: type("E", (), {"embed_one": staticmethod(lambda t: [0.0] * 1536)})(),
    )
    monkeypatch.setattr(deps, "get_token_encoder", lambda: CharEncoder())
    monkeypatch.setattr(deps, "get_idempotency_store", lambda: store)
    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: type("S", (), {"RETRIEVAL_API_KEY": "r", "ESTIMATE_API_KEY": _KEY})(),
    )
    return calls


def test_repeated_idempotency_key_serves_cache_without_calling_llm(generate_calls):
    client = TestClient(app)
    headers = {"X-API-Key": _KEY}
    body = {"transcript": "x" * 200, "idempotency_key": "abc-123"}

    first = client.post("/v1/estimate/from-transcript", json=body, headers=headers)
    second = client.post("/v1/estimate/from-transcript", json=body, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert generate_calls["generate"] == 1


def test_distinct_idempotency_keys_run_the_pipeline_each_time(generate_calls):
    client = TestClient(app)
    headers = {"X-API-Key": _KEY}

    client.post(
        "/v1/estimate/from-transcript",
        json={"transcript": "x" * 200, "idempotency_key": "key-A"},
        headers=headers,
    )
    client.post(
        "/v1/estimate/from-transcript",
        json={"transcript": "x" * 200, "idempotency_key": "key-B"},
        headers=headers,
    )

    assert generate_calls["generate"] == 2
