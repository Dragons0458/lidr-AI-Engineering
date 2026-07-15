"""Instrumentation tests — spans without sending telemetry."""

from __future__ import annotations

from contextlib import contextmanager

import logfire
import pytest

from app.generation.agentic.agent_schemas import (
    AgentModuleNode,
    AgentStructure,
    AgentTaskNode,
    SearchBudgetsArgs,
)
from app.generation.agentic.graph.build import build_estimation_graph
from app.generation.agentic.graph.nodes import GraphNodeDeps
from app.generation.rag.schemas import EstimationQuery

logfire.configure(send_to_logfire=False)


@pytest.mark.asyncio
async def test_graph_emits_five_node_spans(monkeypatch):
    spans: list[str] = []

    @contextmanager
    def fake_span(name, **kwargs):
        spans.append(name)
        yield

    monkeypatch.setattr(logfire, "span", fake_span)

    async def reformulate(transcript: str) -> EstimationQuery:
        return EstimationQuery(function="Portal")

    async def propose_structure(brief: EstimationQuery) -> AgentStructure:
        return AgentStructure(
            modules=[
                AgentModuleNode(
                    name="Core",
                    tasks=[AgentTaskNode(name="Auth", description="")],
                )
            ],
            confidence="high",
            reasoning="ok",
        )

    async def backend(args: SearchBudgetsArgs) -> list[dict]:
        return [
            {
                "id": 1,
                "budget_id": "B1",
                "estimated_hours": 40,
                "distance": 0.1,
            }
        ]

    graph = build_estimation_graph(
        GraphNodeDeps(
            reformulate=reformulate,
            propose_structure=propose_structure,
            retrieval_backend=backend,
        )
    )
    await graph.ainvoke(
        {"transcript": "portal"},
        config={"configurable": {"thread_id": "span-1"}},
    )
    node_spans = [
        s for s in spans if s.startswith("agent.graph.") and ".component" not in s
    ]
    assert node_spans == [
        "agent.graph.extract_requirements",
        "agent.graph.classify_components",
        "agent.graph.search_budgets",
        "agent.graph.generate_estimate",
        "agent.graph.validate_and_consolidate",
    ]
