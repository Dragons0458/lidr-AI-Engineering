"""Unit tests for line-level citation verification (Session 11)."""

from __future__ import annotations

import pytest

from app.generation.rag.context_assembler import build_context_block
from app.generation.rag.schemas import (
    Estimate,
    RetrievedChunk,
    SourceReference,
    TaskItem,
    WorkModule,
)
from app.generation.rag.validation import verify_citations


def _chunk(
    chunk_id: int,
    *,
    budget_id: str = "S07-ECO-001",
    content: str = "Stripe checkout integration — estimated hours: 140",
) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content=content,
        sector="ecommerce",
        project_year=2024,
        chunk_type="budget_component",
        distance=0.3,
        budget_id=budget_id,
        source_id=budget_id,
    )


def _source(
    chunk_id: int, *, document_id: str = "S07-ECO-001", evidence: str
) -> SourceReference:
    return SourceReference(
        chunk_id=chunk_id,
        document_id=document_id,
        evidence=evidence,
    )


def _task(
    *,
    name: str = "Checkout",
    grounded: bool = True,
    sources: list[SourceReference] | None = None,
) -> TaskItem:
    return TaskItem(
        name=name,
        engineer_days=18,
        grounded=grounded,
        sources=sources or [],
    )


def _estimate(*tasks: TaskItem) -> Estimate:
    return Estimate(
        total_engineer_days=18,
        confidence="high",
        reasoning="test",
        modules=[WorkModule(name="Payments", tasks=list(tasks))],
    )


def test_verify_citations_flags_invented_chunk_id_in_dangling():
    chunks = [_chunk(1), _chunk(2)]
    estimate = _estimate(
        _task(
            sources=[
                _source(99, evidence="invented id not in context"),
            ]
        )
    )
    report = verify_citations(estimate, chunks)
    assert len(report.dangling) == 1
    assert report.dangling[0].cited_chunk_ids == [99]
    assert report.grounded == []
    assert report.insufficient == []


def test_verify_citations_grounded_with_valid_chunk_is_grounded():
    chunks = [_chunk(1)]
    estimate = _estimate(_task(sources=[_source(1, evidence="estimated hours: 140")]))
    report = verify_citations(estimate, chunks)
    assert len(report.grounded) == 1
    assert report.dangling == []
    assert report.insufficient == []


def test_verify_citations_grounded_false_is_insufficient():
    chunks = [_chunk(1)]
    estimate = _estimate(_task(grounded=False, sources=[]))
    report = verify_citations(estimate, chunks)
    assert len(report.insufficient) == 1
    assert report.grounded == []
    assert report.dangling == []


def test_task_item_grounded_true_without_sources_raises():
    with pytest.raises(ValueError, match="grounded=True requires"):
        TaskItem(name="Auth", grounded=True, sources=[])


def test_context_block_includes_document_id():
    block = build_context_block([_chunk(7, budget_id="S07-HLT-001")])
    assert 'document_id="S07-HLT-001"' in block
    assert 'id="7"' in block
