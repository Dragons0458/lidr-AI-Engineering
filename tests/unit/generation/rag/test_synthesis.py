"""Unit tests for hour-range synthesis (Session 11)."""

from __future__ import annotations

import pytest

import app.generation.rag.task_hours as th
from app.generation.rag.quality.synthesis import is_contradiction, synthesize_range
from app.generation.rag.schemas import RetrievalResult, RetrievedChunk, TaskNeighbor


def test_is_contradiction_respects_threshold_and_none():
    assert is_contradiction(None, 0.35) is False
    assert is_contradiction(0.34, 0.35) is False
    assert is_contradiction(0.35, 0.35) is True


@pytest.mark.asyncio
async def test_synthesize_range_returns_none_without_contradiction():
    neighbors = [
        TaskNeighbor(source_id=1, estimated_hours=40, distance=0.1),
        TaskNeighbor(source_id=2, estimated_hours=42, distance=0.12),
    ]
    result = await synthesize_range(neighbors, 0.1, threshold=0.35, use_llm=False)
    assert result is None


@pytest.mark.asyncio
async def test_synthesize_range_deterministic_min_max():
    neighbors = [
        TaskNeighbor(source_id=1, budget_id="A", estimated_hours=40, distance=0.1),
        TaskNeighbor(source_id=2, budget_id="B", estimated_hours=90, distance=0.2),
    ]
    result = await synthesize_range(neighbors, 0.5, threshold=0.35, use_llm=False)
    assert result is not None
    assert result.low == 40
    assert result.high == 90
    assert result.reason


def _chunk(cid: int, hours: int, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid,
        content=f"task ~{hours}h",
        chunk_type="historical_task",
        distance=distance,
        estimated_hours=hours,
    )


@pytest.mark.asyncio
async def test_estimate_one_with_synthesis_fills_hours_range(monkeypatch):
    async def fake_retrieve(**kwargs):
        return RetrievalResult(
            chunks=[
                _chunk(1, 40, 0.05),
                _chunk(2, 90, 0.08),
            ],
            low_confidence=False,
            candidates_evaluated=2,
        )

    monkeypatch.setattr(
        "app.dependencies.get_embedder",
        lambda: type("E", (), {"embed_one": staticmethod(lambda t: [0.0] * 1536)})(),
    )
    monkeypatch.setattr(th, "retrieve", fake_retrieve)

    estimate = await th.estimate_one(
        "Notifications",
        "Real-time notifications delivery service",
        "Multichannel alerts",
        top_k=5,
        distance_threshold=0.45,
        synthesis=True,
        contradiction_threshold=0.35,
    )
    assert estimate.has_match is True
    assert estimate.hours_range is not None
    assert estimate.hours_range.low == 40
    assert estimate.hours_range.high == 90
