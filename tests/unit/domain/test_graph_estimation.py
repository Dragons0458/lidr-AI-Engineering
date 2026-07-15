"""Unit tests for the Session 13 graph estimation conductor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.domain.graph_estimation import (
    GraphEstimationError,
    GraphRuntimeUnavailableError,
    run_graph_estimation,
)
from app.domain.schemas.graph_estimation import GraphEstimationRequest
from app.generation.agentic.agent_schemas import AgentEstimate


@pytest.mark.asyncio
async def test_run_graph_estimation_requires_runtime():
    with pytest.raises(GraphRuntimeUnavailableError):
        await run_graph_estimation(
            estimation_id="e1",
            transcript="hello",
            runtime=None,
        )


@pytest.mark.asyncio
async def test_run_graph_estimation_maps_public_response():
    estimate = AgentEstimate(
        components=[
            {
                "name": "Auth",
                "estimated_hours": 46.0,
                "cited_chunk_ids": [1],
                "rationale": "median + contingency",
            }
        ],
        total_hours=46.0,
        assumptions=[],
        confidence="high",
    )
    graph = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value={
                "estimate": estimate.model_dump(mode="json"),
                "status": "validated",
            }
        )
    )
    runtime = SimpleNamespace(graph=graph)
    response = await run_graph_estimation(
        estimation_id="est-42",
        transcript="Build portal",
        runtime=runtime,
    )
    graph.ainvoke.assert_awaited_once_with(
        {"transcript": "Build portal"},
        config={"configurable": {"thread_id": "est-42"}},
    )
    assert response.status == "validated"
    assert response.estimate.total_hours == 46.0


@pytest.mark.asyncio
async def test_run_graph_estimation_incomplete_result_is_technical_error():
    graph = SimpleNamespace(ainvoke=AsyncMock(return_value={"status": "validated"}))
    runtime = SimpleNamespace(graph=graph)
    with pytest.raises(GraphEstimationError):
        await run_graph_estimation(
            estimation_id="e1",
            transcript="x",
            runtime=runtime,
        )


def test_request_schema_requires_estimation_id_and_transcript():
    with pytest.raises(Exception):
        GraphEstimationRequest.model_validate({"transcript": "only"})
    with pytest.raises(Exception):
        GraphEstimationRequest.model_validate({"estimation_id": "e1"})
    ok = GraphEstimationRequest(estimation_id="e1", transcript="hello")
    assert ok.estimation_id == "e1"
