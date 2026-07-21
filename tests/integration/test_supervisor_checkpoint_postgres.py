"""Postgres integration for Session 14 supervisor checkpoint pause/resume.

Marked so default unit suites stay network/DB free.
"""

from __future__ import annotations

import os
import uuid

import pytest
from langgraph.types import Command

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_supervisor_pause_survives_reinstantiation():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not set")

    from app.foundation.persistence.langgraph import (
        close_langgraph_runtime,
        open_langgraph_runtime,
    )
    from app.generation.agentic.agent_schemas import (
        AgentModuleNode,
        AgentStructure,
        AgentTaskNode,
    )
    from app.generation.agentic.graph.supervisor_build import build_supervisor_graph
    from app.generation.agentic.graph.supervisor_nodes import (
        SupervisorDecision,
        SupervisorDeps,
    )
    from app.generation.rag.schemas import EstimationQuery

    async def reformulate(_t: str) -> EstimationQuery:
        return EstimationQuery(function="exotic", technologies=[])

    async def propose_structure(_b: EstimationQuery) -> AgentStructure:
        return AgentStructure(
            modules=[
                AgentModuleNode(
                    name="QKD",
                    tasks=[AgentTaskNode(name="Link", description="quantum")],
                )
            ],
            confidence="low",
            reasoning="edge",
        )

    async def route(_d: str) -> SupervisorDecision:
        raise RuntimeError("stub")

    async def backend(_a):
        return []

    deps = SupervisorDeps(
        reformulate=reformulate,
        propose_structure=propose_structure,
        retrieval_backend=backend,
        route_with_model=route,
        confidence_threshold=0.99,
        min_grounded_ratio=0.99,
        out_of_range_factor=2.0,
        max_steps=8,
        privilege_strict=False,
    )

    estimation_id = f"s14-it-{uuid.uuid4().hex[:8]}"
    thread = {"configurable": {"thread_id": f"s14:{estimation_id}"}}

    runtime1 = await open_langgraph_runtime(
        database_url,
        build_graph=lambda checkpointer: build_supervisor_graph(
            deps, checkpointer=checkpointer
        ),
    )
    try:
        await runtime1.graph.ainvoke(
            {
                "transcript": (
                    "Quantum key distribution over COBOL mainframe iris HSM. " * 8
                ),
                "estimation_id": estimation_id,
            },
            thread,
        )
        snap1 = await runtime1.graph.aget_state(thread)
        assert snap1.next
        assert snap1.interrupts
    finally:
        await close_langgraph_runtime(runtime1)

    runtime2 = await open_langgraph_runtime(
        database_url,
        build_graph=lambda checkpointer: build_supervisor_graph(
            deps, checkpointer=checkpointer
        ),
    )
    try:
        snap2 = await runtime2.graph.aget_state(thread)
        assert snap2.next
        await runtime2.graph.ainvoke(
            Command(resume={"decision": "approve", "note": "it"}),
            thread,
        )
        snap3 = await runtime2.graph.aget_state(thread)
        assert not snap3.next
        assert snap3.values.get("status") == "validated"
        # Keyed reducers must not duplicate routing steps.
        steps = [r.get("step") for r in (snap3.values.get("routing_history") or [])]
        assert len(steps) == len(set(steps))
    finally:
        await close_langgraph_runtime(runtime2)
