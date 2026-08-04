"""Unit tests for the Session 14 supervisor domain conductor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.schemas.supervisor_estimation import SupervisorRunState
from app.domain.supervisor_estimation import (
    SupervisorConflictError,
    SupervisorNotFoundError,
    SupervisorRuntimeUnavailableError,
    read_supervisor_state,
    resume_supervisor_run,
    snapshot_to_run_state,
    start_supervisor_run,
)


def _snapshot(*, paused: bool = False, with_interrupt: bool = False):
    values = {
        "estimate": {"total_hours": 10, "components": []},
        "confidence": 0.4,
        "requirements": ["r1"],
        "components": [],
        "budget_matches": [],
        "validation": {"ok": True, "issues": []},
        "status": "needs_review",
        "routing_history": [
            {"step": 0, "next_agent": "finish", "reason": "done", "source": "llm"}
        ],
        "agent_contributions": [
            {
                "outcome": "denied",
                "agent": "budget_searcher",
                "tool": "validate_estimate",
            }
        ],
        "errors": [],
    }
    interrupts = ()
    if with_interrupt:
        interrupts = (
            SimpleNamespace(
                value={
                    "gate": "low_confidence_review",
                    "reasons": ["low_confidence"],
                    "confidence": 0.4,
                    "threshold": 0.6,
                    "estimate": values["estimate"],
                    "validation": values["validation"],
                }
            ),
        )
    return SimpleNamespace(
        values=values,
        next=("human_review_gate",) if paused else (),
        interrupts=interrupts,
        created_at="2026-01-01",
    )


def test_snapshot_derives_awaiting_human_review():
    run = snapshot_to_run_state("e1", _snapshot(paused=True, with_interrupt=True))
    assert run.status == "awaiting_human_review"
    assert run.state == "paused"
    assert run.pending_review is not None
    assert run.privilege_violations


def test_snapshot_completed_keeps_stored_status():
    run = snapshot_to_run_state("e1", _snapshot(paused=False))
    assert run.status == "needs_review"
    assert run.state == "completed"
    assert run.pending_review is None


@pytest.mark.asyncio
async def test_start_requires_supervisor_graph():
    runtime = SimpleNamespace(supervisor_graph=None)
    with pytest.raises(SupervisorRuntimeUnavailableError):
        await start_supervisor_run("x" * 120, runtime)


@pytest.mark.asyncio
async def test_start_invokes_graph():
    graph = MagicMock()
    graph.ainvoke = AsyncMock()
    graph.aget_state = AsyncMock(return_value=_snapshot())
    runtime = SimpleNamespace(supervisor_graph=graph)
    result = await start_supervisor_run("x" * 120, runtime, estimation_id="est-9")
    assert isinstance(result, SupervisorRunState)
    assert result.estimation_id == "est-9"
    graph.ainvoke.assert_awaited_once()
    kwargs = graph.ainvoke.await_args.kwargs
    args = graph.ainvoke.await_args.args
    config = kwargs.get("config") if kwargs else None
    if config is None and len(args) > 1:
        config = args[1]
    assert config["configurable"]["thread_id"] == "s14:est-9"


@pytest.mark.asyncio
async def test_resume_conflict_when_not_paused():
    graph = MagicMock()
    graph.aget_state = AsyncMock(return_value=_snapshot(paused=False))
    runtime = SimpleNamespace(supervisor_graph=graph)
    with pytest.raises(SupervisorConflictError):
        await resume_supervisor_run("e1", {"decision": "approve"}, runtime)


@pytest.mark.asyncio
async def test_read_not_found():
    graph = MagicMock()
    graph.aget_state = AsyncMock(
        return_value=SimpleNamespace(values={}, next=(), interrupts=(), created_at=None)
    )
    runtime = SimpleNamespace(supervisor_graph=graph)
    with pytest.raises(SupervisorNotFoundError):
        await read_supervisor_state("missing", runtime)
