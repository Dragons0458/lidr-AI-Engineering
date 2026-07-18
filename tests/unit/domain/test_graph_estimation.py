"""Unit tests for the Session 13 graph estimation domain conductor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langgraph.types import Command

from app.domain.graph_estimation import (
    GraphConflictError,
    GraphEstimationError,
    GraphNotFoundError,
    GraphRuntimeUnavailableError,
    draft_graph_proposal,
    progress_state,
    resume_graph_run,
    snapshot_to_run_state,
    start_graph_run,
)
from app.domain.schemas.graph_estimation import GraphEstimationRequest
from app.generation.agentic.graph.schemas import CommercialProposal


def _snapshot(*, paused: bool, values=None, interrupts=()):
    return SimpleNamespace(
        values=values or {"complexity": "high"},
        next=("human_gate_structure",) if paused else (),
        interrupts=interrupts,
        created_at="now" if values else None,
    )


def test_request_schema_requires_estimation_id_and_transcript():
    with pytest.raises(Exception):
        GraphEstimationRequest.model_validate({"transcript": "only"})
    with pytest.raises(Exception):
        GraphEstimationRequest.model_validate({"estimation_id": "e1"})
    ok = GraphEstimationRequest(estimation_id="e1", transcript="hello")
    assert ok.estimation_id == "e1"


def test_snapshot_to_run_state_paused_with_pending_gate():
    interrupt = SimpleNamespace(
        value={
            "gate": "structure_review",
            "estimation_id": "e1",
            "structure": {"modules": []},
        }
    )
    state = snapshot_to_run_state(
        "e1",
        _snapshot(paused=True, interrupts=(interrupt,)),
    )
    assert state.state == "paused"
    assert state.pending_gate is not None
    assert state.pending_gate.gate == "structure_review"
    assert "structure" in state.pending_gate.payload


def test_snapshot_to_run_state_completed():
    state = snapshot_to_run_state(
        "e1",
        _snapshot(
            paused=False, values={"status": "validated", "estimate": {"modules": []}}
        ),
    )
    assert state.state == "completed"
    assert state.pending_gate is None
    assert state.status == "validated"


def test_progress_state_running_vs_paused():
    running = _snapshot(paused=True, interrupts=())
    paused = _snapshot(
        paused=True,
        interrupts=(SimpleNamespace(value={"gate": "structure_review"}),),
    )
    assert progress_state(running) == "running"
    assert progress_state(paused) == "paused"


@pytest.mark.asyncio
async def test_start_graph_run_requires_runtime():
    with pytest.raises(GraphRuntimeUnavailableError):
        await start_graph_run("e1", "hello", None)


@pytest.mark.asyncio
async def test_start_graph_run_maps_snapshot():
    graph = SimpleNamespace(
        ainvoke=AsyncMock(return_value={}),
        aget_state=AsyncMock(
            return_value=_snapshot(
                paused=True,
                interrupts=(SimpleNamespace(value={"gate": "structure_review"}),),
            )
        ),
    )
    runtime = SimpleNamespace(graph=graph)
    state = await start_graph_run("e1", "hello", runtime)
    graph.ainvoke.assert_awaited_once_with(
        {"transcript": "hello", "estimation_id": "e1"},
        config={"configurable": {"thread_id": "e1"}},
    )
    assert state.state == "paused"


@pytest.mark.asyncio
async def test_resume_graph_run_conflict_when_not_paused():
    graph = SimpleNamespace(
        aget_state=AsyncMock(return_value=_snapshot(paused=False)),
    )
    runtime = SimpleNamespace(graph=graph)
    with pytest.raises(GraphConflictError):
        await resume_graph_run("e1", {"approved": True}, runtime)


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_read_graph_state_not_found():
    graph = SimpleNamespace(
        aget_state=AsyncMock(
            return_value=SimpleNamespace(
                values=None,
                next=(),
                interrupts=(),
                created_at=None,
            )
        ),
    )
    runtime = SimpleNamespace(graph=graph)
    with pytest.raises(GraphNotFoundError):
        from app.domain.graph_estimation import read_graph_state

        await read_graph_state("missing", runtime)


@pytest.mark.asyncio
async def test_resume_graph_run_uses_command_resume():
    graph = SimpleNamespace(
        aget_state=AsyncMock(
            return_value=_snapshot(
                paused=True,
                interrupts=(SimpleNamespace(value={"gate": "structure_review"}),),
            )
        ),
        ainvoke=AsyncMock(return_value={}),
    )
    graph.aget_state.side_effect = [
        graph.aget_state.return_value,
        _snapshot(paused=False, values={"status": "validated"}),
    ]
    runtime = SimpleNamespace(graph=graph)
    state = await resume_graph_run("e1", {"approved": True}, runtime)
    graph.ainvoke.assert_awaited_once()
    assert isinstance(graph.ainvoke.await_args.args[0], Command)
    assert state.state == "completed"


@pytest.mark.asyncio
async def test_draft_graph_proposal_conflict_without_estimate():
    graph = SimpleNamespace(
        aget_state=AsyncMock(return_value=_snapshot(paused=False, values={})),
    )
    runtime = SimpleNamespace(graph=graph)

    async def propose(_: str) -> CommercialProposal:
        return CommercialProposal(
            title="T",
            executive_summary="S",
            body_markdown="#",
        )

    with pytest.raises(GraphConflictError):
        await draft_graph_proposal("e1", runtime, propose=propose)


@pytest.mark.asyncio
async def test_start_graph_run_wraps_failures():
    graph = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("boom")))
    runtime = SimpleNamespace(graph=graph)
    with pytest.raises(GraphEstimationError):
        await start_graph_run("e1", "hello", runtime)
