"""Topology and execution-order tests for the Session 13 StateGraph."""

from __future__ import annotations

import logfire
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.generation.agentic.agent_schemas import (
    AgentModuleNode,
    AgentStructure,
    AgentTaskNode,
    SearchBudgetsArgs,
)
from app.generation.agentic.graph.build import NODE_NAMES, build_sequential_graph
from app.generation.agentic.graph.nodes import GraphNodeDeps
from app.generation.rag.schemas import EstimationQuery

logfire.configure(send_to_logfire=False)


def _deps(order: list[str]) -> GraphNodeDeps:
    async def reformulate(transcript: str) -> EstimationQuery:
        order.append("extract_requirements")
        return EstimationQuery(function="Portal", technologies=["Python"])

    async def propose_structure(brief: EstimationQuery) -> AgentStructure:
        order.append("classify_components")
        return AgentStructure(
            modules=[
                AgentModuleNode(
                    name="Core",
                    tasks=[AgentTaskNode(name="Auth", description="login")],
                )
            ],
            confidence="high",
            reasoning="ok",
        )

    async def backend(args: SearchBudgetsArgs) -> list[dict]:
        order.append("search_budgets")
        return [
            {
                "id": 1,
                "budget_id": "B1",
                "estimated_hours": 40,
                "distance": 0.2,
            }
        ]

    return GraphNodeDeps(
        reformulate=reformulate,
        propose_structure=propose_structure,
        retrieval_backend=backend,
    )


@pytest.mark.asyncio
async def test_graph_runs_five_nodes_in_mandatory_order():
    order: list[str] = []
    graph = build_sequential_graph(_deps(order), checkpointer=InMemorySaver())
    result = await graph.ainvoke(
        {"transcript": "Build a portal"},
        config={"configurable": {"thread_id": "topo-1"}},
    )
    assert order == [
        "extract_requirements",
        "classify_components",
        "search_budgets",
    ]
    assert result["status"] in {"validated", "needs_review"}
    assert result["estimate"]["total_hours"] > 0
    assert list(NODE_NAMES) == [
        "extract_requirements",
        "classify_components",
        "search_budgets",
        "generate_estimate",
        "validate_and_consolidate",
    ]


@pytest.mark.asyncio
async def test_graph_exposes_estimate_and_status():
    order: list[str] = []
    graph = build_sequential_graph(_deps(order))
    result = await graph.ainvoke(
        {"transcript": "Build a portal"},
        config={"configurable": {"thread_id": "topo-2"}},
    )
    assert "estimate" in result
    assert result["status"] == "validated"
    assert len(result["components"]) == 1
