"""Unit tests for supervisor graph topology and a happy offline run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.generation.agentic.agent_schemas import (
    AgentModuleNode,
    AgentStructure,
    AgentTaskNode,
)
from app.generation.agentic.graph.supervisor_build import (
    AGENT_NODE_NAMES,
    build_supervisor_graph,
)
from app.generation.agentic.graph.supervisor_nodes import (
    SupervisorDecision,
    SupervisorDeps,
)
from app.generation.rag.schemas import EstimationQuery

_SEQUENCE = [
    "requirements_extractor",
    "budget_searcher",
    "estimate_generator",
    "coherence_validator",
    "finish",
]


async def _reformulate(_t: str) -> EstimationQuery:
    return EstimationQuery(
        function="portal",
        technologies=["python"],
        sector="logistics",
        scale="small",
        country="ES",
        regulations=[],
        constraints=[],
    )


async def _structure(_b: EstimationQuery) -> AgentStructure:
    return AgentStructure(
        modules=[
            AgentModuleNode(
                name="Backend",
                tasks=[AgentTaskNode(name="API", description="REST")],
            )
        ],
        confidence="high",
        reasoning="one clear module",
    )


async def _backend(_args: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": 11,
            "budget_id": "b11",
            "estimated_hours": 32.0,
            "distance": 0.05,
            "content": "api",
        }
    ]


async def _route(_digest: str) -> SupervisorDecision:
    # The supervisor's legality + fallback will keep us honest; return the next
    # unused ladder step based on a simple counter stored on the function.
    idx = getattr(_route, "_idx", 0)
    target = _SEQUENCE[min(idx, len(_SEQUENCE) - 1)]
    _route._idx = idx + 1  # type: ignore[attr-defined]
    return SupervisorDecision(
        next_agent=target, reason=f"step {idx}", confidence="high"
    )


def _deps() -> SupervisorDeps:
    _route._idx = 0  # type: ignore[attr-defined]
    return SupervisorDeps(
        reformulate=_reformulate,
        propose_structure=_structure,
        retrieval_backend=_backend,
        route_with_model=_route,
        confidence_threshold=0.1,
        min_grounded_ratio=0.1,
        out_of_range_factor=10.0,
        max_steps=8,
        privilege_strict=False,
    )


def test_topology_has_star_shape():
    graph = build_supervisor_graph(_deps(), checkpointer=MemorySaver())
    drawable = graph.get_graph()
    node_ids = set(drawable.nodes)
    for name in AGENT_NODE_NAMES:
        assert name in node_ids
    assert "supervisor" in node_ids
    assert "human_review_gate" in node_ids


@pytest.mark.asyncio
async def test_happy_path_completes_with_fakes():
    graph = build_supervisor_graph(_deps(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "s14:test-happy"}}
    await graph.ainvoke(
        {
            "transcript": "We need a small supplier portal with a REST API. " * 5,
            "estimation_id": "test-happy",
        },
        config,
    )
    snap = await graph.aget_state(config)
    assert not snap.next
    values = snap.values
    assert values.get("estimate")
    assert values.get("routing_history")
    assert any(
        r.get("next_agent") == "requirements_extractor"
        for r in values["routing_history"]
    )


@pytest.mark.asyncio
async def test_edge_scope_pauses_even_with_close_historical_matches():
    graph = build_supervisor_graph(_deps(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "s14:test-edge-risk"}}
    transcript = Path("exercises/session-14/sample_transcript_edge_case.txt").read_text(
        encoding="utf-8"
    )

    await graph.ainvoke(
        {"transcript": transcript, "estimation_id": "test-edge-risk"},
        config,
    )
    snap = await graph.aget_state(config)

    assert snap.next == ("human_review_gate",)
    assert snap.interrupts
    assert "high_risk_scope" in snap.interrupts[0].value["reasons"]
    assert "novel_cryptography" in snap.interrupts[0].value["risk_flags"]
