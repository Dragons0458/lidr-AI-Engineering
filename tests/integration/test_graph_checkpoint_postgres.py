"""Integration: multi-agent graph pauses on Postgres checkpointer."""

from __future__ import annotations

import uuid

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.config import get_settings
from app.foundation.persistence.langgraph import (
    close_langgraph_runtime,
    open_langgraph_runtime,
    to_libpq_conninfo,
)
from app.generation.agentic.agent_schemas import (
    AgentModuleNode,
    AgentStructure,
    AgentTaskNode,
)
from app.generation.agentic.graph.agent_nodes import MultiAgentDeps
from app.generation.agentic.graph.build import build_estimation_graph
from app.generation.agentic.graph.schemas import (
    ComplexityClassification,
    ReliabilityReport,
)
from app.generation.rag.schemas import TaskHoursEstimate

pytestmark = pytest.mark.integration

TRANSCRIPT = "Build a portal with authentication and reporting modules."


def _deps() -> MultiAgentDeps:
    async def classify(transcript: str) -> ComplexityClassification:
        return ComplexityClassification(
            complexity="medium",
            reformulated_transcript=transcript,
            reasoning="ok",
        )

    async def propose_structure(brief: str, reasoning_effort: str) -> AgentStructure:
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

    async def estimate_task(module, name, description):
        return TaskHoursEstimate(
            module=module,
            task=name,
            estimated_hours=40,
            reliability=0.9,
            has_match=True,
            neighbors=[],
        )

    async def analyze(digest: str) -> ReliabilityReport:
        return ReliabilityReport(
            overall_confidence="high",
            grounded_task_ratio=1.0,
            weak_points=[],
            summary="ok",
        )

    async def propose(user_message: str):
        from app.generation.agentic.graph.schemas import CommercialProposal

        return CommercialProposal(
            title="T",
            executive_summary="S",
            body_markdown="# Proposal",
        )

    return MultiAgentDeps(
        classify=classify,
        propose_structure=propose_structure,
        estimate_task=estimate_task,
        recover=None,
        analyze=analyze,
        propose=propose,
        recovery_reliability_threshold=0.35,
        structure_effort_by_complexity={
            "low": "low",
            "medium": "medium",
            "high": "high",
        },
        default_reasoning_effort="medium",
    )


@pytest.mark.asyncio
async def test_multiagent_graph_pauses_on_gate1_with_postgres():
    settings = get_settings()
    assert "postgresql://" in to_libpq_conninfo(settings.DATABASE_URL)

    try:
        runtime = await open_langgraph_runtime(
            settings.DATABASE_URL,
            build_graph=lambda checkpointer: build_estimation_graph(
                _deps(), checkpointer=checkpointer, proposal_enabled=False
            ),
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres unavailable: {exc}")

    thread_id = f"s13-multi-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    try:
        await runtime.graph.ainvoke(
            {"transcript": TRANSCRIPT, "estimation_id": thread_id},
            config=config,
        )
        snap = await runtime.graph.aget_state(config)
        assert snap.next == ("human_gate_structure",)
        assert snap.interrupts[0].value["gate"] == "structure_review"

        await runtime.graph.ainvoke(Command(resume={"approved": True}), config)
        snap = await runtime.graph.aget_state(config)
        assert snap.next == ("human_gate_analysis",)

        result = await runtime.graph.ainvoke(
            Command(resume={"validated": True, "want_proposal": False}),
            config,
        )
        assert result["status"] == "validated"
        final = await runtime.graph.aget_state(config)
        assert final.next == ()
    finally:
        await close_langgraph_runtime(runtime)


@pytest.mark.asyncio
async def test_memory_saver_multiagent_pause_resume():
    graph = build_estimation_graph(
        _deps(), checkpointer=MemorySaver(), proposal_enabled=False
    )
    thread = f"mem-multi-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread}}
    await graph.ainvoke(
        {"transcript": TRANSCRIPT, "estimation_id": thread},
        config=config,
    )
    snap = await graph.aget_state(config)
    assert snap.next == ("human_gate_structure",)
