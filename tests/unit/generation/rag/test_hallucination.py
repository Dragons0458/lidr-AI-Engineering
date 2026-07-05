"""Unit tests for the semantic hallucination gate (Session 11)."""

from __future__ import annotations

import pytest

from app.generation.rag.quality.hallucination import (
    gate_estimate,
    gate_line,
    numeric_anchor,
)
from app.generation.rag.schemas import (
    Estimate,
    LineVerdict,
    RetrievedChunk,
    SourceReference,
    TaskItem,
    WorkModule,
)


def _chunk(cid: int, content: str, *, hours: int | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid,
        content=content,
        chunk_type="budget_component",
        distance=0.1,
        estimated_hours=hours,
    )


def _task(
    name: str,
    *,
    days: int | None,
    grounded: bool,
    chunk_id: int = 1,
) -> TaskItem:
    sources = (
        [SourceReference(chunk_id=chunk_id, document_id="B-1", evidence="40 hours")]
        if grounded
        else []
    )
    return TaskItem(
        name=name,
        description="scope",
        engineer_days=days,
        grounded=grounded,
        sources=sources,
    )


def test_numeric_anchor_from_metadata_and_regex():
    chunk_meta = _chunk(1, "ignored", hours=80)
    task = _task("Auth", days=10, grounded=True, chunk_id=1)
    assert numeric_anchor(task, {1: chunk_meta}) == pytest.approx(10.0)

    chunk_regex = _chunk(2, "Payments integration: 16 hours")
    task2 = _task("Pay", days=2, grounded=True, chunk_id=2)
    assert numeric_anchor(task2, {2: chunk_regex}) == pytest.approx(2.0)


def test_gate_line_grounded():
    task = _task("Auth", days=5, grounded=True)
    gate = gate_line(task, "Platform", anchor_days=5.0, verdict=None, tolerance=0.5)
    assert gate.status == "grounded"


def test_gate_line_degraded_on_overage():
    task = _task("Auth", days=20, grounded=True)
    gate = gate_line(task, "Platform", anchor_days=5.0, verdict=None, tolerance=0.5)
    assert gate.status == "degraded"
    assert gate.numeric_deviation is not None


def test_gate_line_insufficient_when_not_grounded():
    task = _task("Auth", days=5, grounded=False)
    gate = gate_line(task, "Platform", anchor_days=5.0, verdict=None, tolerance=0.5)
    assert gate.status == "insufficient"


def test_gate_line_degraded_on_judge_fail():
    task = _task("Auth", days=5, grounded=True)
    verdict = LineVerdict(
        module="Platform", component="Auth", entailed=False, reason="unsupported"
    )
    gate = gate_line(task, "Platform", anchor_days=5.0, verdict=verdict, tolerance=0.5)
    assert gate.status == "degraded"


@pytest.mark.asyncio
async def test_gate_estimate_aggregates_without_judge():
    estimate = Estimate(
        confidence="high",
        reasoning="r",
        modules=[
            WorkModule(
                name="Platform",
                tasks=[
                    _task("Auth", days=20, grounded=True),
                    _task("Spare", days=None, grounded=False),
                ],
            )
        ],
    )
    chunks = [_chunk(1, "Auth :: 40 hours", hours=40)]
    report = await gate_estimate(
        estimate,
        chunks,
        tolerance=0.5,
        judge_model="gpt-4o-mini",
        use_judge=False,
    )
    assert report.total_lines == 2
    assert report.degraded_lines == 1
    assert report.insufficient_lines == 1
