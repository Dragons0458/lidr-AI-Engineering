"""Keyed ``merge_task_hours`` reducer semantics (Session 13 live)."""

from __future__ import annotations

import operator
import typing

from app.generation.agentic.graph.state import EstimationState, merge_task_hours


def test_task_hours_reducer_is_keyed_not_operator_add():
    hints = typing.get_type_hints(EstimationState, include_extras=True)
    metadata = getattr(hints["task_hours"], "__metadata__", ())
    assert merge_task_hours in metadata
    assert operator.add not in metadata


def test_merge_task_hours_dedupes_by_module_and_task():
    existing = [
        {"module": "Backend", "task": "API", "estimated_hours": 40},
        {"module": "Backend", "task": "Auth", "estimated_hours": 20},
    ]
    new = [
        {"module": "Backend", "task": "API", "estimated_hours": 64, "has_match": True}
    ]
    merged = merge_task_hours(existing, new)
    by_task = {(t["module"], t["task"]): t["estimated_hours"] for t in merged}
    assert by_task == {("Backend", "API"): 64, ("Backend", "Auth"): 20}
    assert len(merged) == 2


def test_merge_task_hours_is_idempotent_for_same_batch():
    batch = [{"module": "M", "task": "T", "estimated_hours": 8}]
    once = merge_task_hours(None, batch)
    twice = merge_task_hours(once, batch)
    assert twice == once


def test_merge_task_hours_handles_empty_sides():
    assert merge_task_hours(
        None, [{"module": "M", "task": "T", "estimated_hours": 8}]
    ) == [{"module": "M", "task": "T", "estimated_hours": 8}]
    assert merge_task_hours([{"module": "M", "task": "T"}], None) == [
        {"module": "M", "task": "T"}
    ]
