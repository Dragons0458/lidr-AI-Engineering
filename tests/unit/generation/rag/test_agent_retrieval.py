"""Unit tests for the recovery-agent retrieval adapter."""

from types import SimpleNamespace

import pytest

import app.generation.rag.agent_retrieval as ar
from app.generation.rag.retrieval.collections import Collection
from app.generation.rag.schemas import RetrievalResult, RetrievedChunk


def _chunk(
    ident: int,
    hours: int | None,
    content: str = "A historical task",
) -> RetrievedChunk:
    return RetrievedChunk(
        id=ident,
        content=content,
        sector="finance",
        chunk_type="historical_task",
        distance=0.123456,
        budget_id="BUD-1",
        estimated_hours=hours,
    )


@pytest.mark.asyncio
async def test_backend_uses_resolved_retrieval_configuration(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        ar,
        "get_embedder",
        lambda: SimpleNamespace(embed_one=lambda query: [0.1, 0.2]),
    )
    monkeypatch.setattr(
        ar,
        "get_settings",
        lambda: SimpleNamespace(RETRIEVAL_RECALL_TOP_K=25, RRF_K=60),
    )

    async def fake_retrieve(**kwargs):
        captured.update(kwargs)
        return RetrievalResult(
            chunks=[_chunk(1, 42)], low_confidence=False, candidates_evaluated=1
        )

    monkeypatch.setattr(ar, "retrieve", fake_retrieve)
    backend = ar.make_retrieval_backend(
        top_k=7,
        distance_threshold=0.45,
        search_mode="hybrid",
        rerank=True,
    )
    result = await backend("payment gateway", ["finance"])

    assert captured["collection"] == Collection.BUDGET
    assert captured["chunk_types"] == ["historical_task"]
    assert captured["top_k"] == 7
    assert captured["distance_threshold"] == 0.45
    assert captured["search_mode"] == "hybrid"
    assert captured["rerank"] is True
    assert captured["sectors"] == ["finance"]
    assert result[0] == {
        "id": 1,
        "content_preview": "A historical task",
        "sector": "finance",
        "budget_id": "BUD-1",
        "estimated_hours": 42,
        "distance": 0.1235,
    }


@pytest.mark.asyncio
async def test_backend_excludes_missing_hours_and_sanitizes_preview(monkeypatch):
    monkeypatch.setattr(
        ar, "get_embedder", lambda: SimpleNamespace(embed_one=lambda query: [0.1])
    )
    monkeypatch.setattr(
        ar,
        "get_settings",
        lambda: SimpleNamespace(RETRIEVAL_RECALL_TOP_K=20, RRF_K=60),
    )

    async def fake_retrieve(**kwargs):
        return RetrievalResult(
            chunks=[
                _chunk(1, None),
                _chunk(2, 10, "  line\x00 one \n " + "x" * 200),
            ],
            low_confidence=False,
            candidates_evaluated=2,
        )

    monkeypatch.setattr(ar, "retrieve", fake_retrieve)
    result = await ar.make_retrieval_backend(
        top_k=5,
        distance_threshold=0.45,
        search_mode="vector",
        rerank=False,
    )("query", None)

    assert [item["id"] for item in result] == [2]
    assert "\x00" not in result[0]["content_preview"]
    assert "\n" not in result[0]["content_preview"]
    assert len(result[0]["content_preview"]) <= 160


@pytest.mark.asyncio
async def test_backend_fails_explicitly_without_embedder(monkeypatch):
    monkeypatch.setattr(ar, "get_embedder", lambda: None)
    backend = ar.make_retrieval_backend(
        top_k=5,
        distance_threshold=0.45,
        search_mode="vector",
        rerank=False,
    )
    with pytest.raises(RuntimeError, match="Embedding service"):
        await backend("query", None)
