"""Unit tests for Session 13 graph state reducers and helpers."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.generation.agentic.graph.state import (
    EstimationState,
    assert_serializable_state,
    ensure_unique_component_ids,
    stable_component_id,
)


@pytest.mark.asyncio
async def test_budget_matches_accumulate_via_reducer():
    async def first(state: EstimationState):
        return {
            "budget_matches": [
                {
                    "component_id": "a",
                    "chunk_id": 1,
                    "reference_budget_id": "b1",
                    "amount": 10.0,
                    "distance": 0.1,
                }
            ]
        }

    async def second(state: EstimationState):
        return {
            "budget_matches": [
                {
                    "component_id": "b",
                    "chunk_id": 2,
                    "reference_budget_id": "b2",
                    "amount": 20.0,
                    "distance": 0.2,
                }
            ]
        }

    builder = StateGraph(EstimationState)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    result = await graph.ainvoke(
        {"transcript": "x"},
        config={"configurable": {"thread_id": "reducer-matches"}},
    )
    assert [m["chunk_id"] for m in result["budget_matches"]] == [1, 2]


@pytest.mark.asyncio
async def test_errors_accumulate_and_status_estimate_overwrite():
    async def seed(state: EstimationState):
        return {
            "errors": ["e1"],
            "status": "needs_review",
            "estimate": {"total_hours": 1},
        }

    async def update(state: EstimationState):
        return {
            "errors": ["e2"],
            "status": "validated",
            "estimate": {"total_hours": 99},
        }

    builder = StateGraph(EstimationState)
    builder.add_node("seed", seed)
    builder.add_node("update", update)
    builder.add_edge(START, "seed")
    builder.add_edge("seed", "update")
    builder.add_edge("update", END)
    graph = builder.compile()
    result = await graph.ainvoke({"transcript": "x"})
    assert result["errors"] == ["e1", "e2"]
    assert result["status"] == "validated"
    assert result["estimate"] == {"total_hours": 99}


def test_stable_component_ids_unique_across_modules():
    left = stable_component_id(0, 0, "Auth")
    right = stable_component_id(1, 0, "Auth")
    assert left != right


def test_ensure_unique_component_ids_suffixes_collisions():
    components = [
        {
            "component_id": "same",
            "name": "A",
            "category": "M1",
            "description": "",
        },
        {
            "component_id": "same",
            "name": "B",
            "category": "M2",
            "description": "",
        },
    ]
    unique = ensure_unique_component_ids(components)
    assert unique[0]["component_id"] == "same"
    assert unique[1]["component_id"] == "same~1"


def test_assert_serializable_state_rejects_clients():
    with pytest.raises(TypeError):
        assert_serializable_state({"client": object()})
