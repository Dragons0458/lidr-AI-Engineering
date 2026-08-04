"""Unit tests for Session 14 competition (divergence + subgraph + penalty)."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.generation.agentic.agent_schemas import (
    AgentModuleNode,
    AgentStructure,
    AgentTaskNode,
)
from app.generation.agentic.graph.schemas import EstimateProposal, SynthesizedEstimate
from app.generation.agentic.graph.supervisor_build import build_supervisor_graph
from app.generation.agentic.graph.supervisor_competition import (
    build_competition_subgraph,
    compute_divergence,
)
from app.generation.agentic.graph.supervisor_nodes import (
    SupervisorDecision,
    SupervisorDeps,
    make_supervisor_nodes,
)
from app.generation.agentic.graph.supervisor_state import apply_divergence_penalty
from app.generation.rag.schemas import EstimationQuery


def test_divergence_is_pure_arithmetic():
    proposals = [
        {"stance": "aggressive", "total_hours": 100},
        {"stance": "conservative", "total_hours": 300},
    ]
    d = compute_divergence(proposals)
    assert d["low"] == 100 and d["high"] == 300
    assert d["spread"] == 200
    assert d["ratio"] == pytest.approx(1.0)
    assert d["level"] == "high"


def test_divergence_levels_are_thresholded():
    close = compute_divergence([{"total_hours": 100}, {"total_hours": 110}])
    assert close["level"] == "low"
    mid = compute_divergence([{"total_hours": 100}, {"total_hours": 140}])
    assert mid["level"] == "medium"


def test_divergence_degrades_gracefully_with_fewer_than_two():
    assert compute_divergence([])["ratio"] == 0.0
    single = compute_divergence([{"total_hours": 80}])
    assert single["low"] == single["high"] == 80
    assert single["ratio"] == 0.0


def test_apply_divergence_penalty():
    assert apply_divergence_penalty(0.9, {"ratio": 0.0}, penalty=0.4) == 0.9
    assert apply_divergence_penalty(0.9, {"ratio": 1.0}, penalty=0.4) == pytest.approx(
        0.5
    )
    assert apply_divergence_penalty(0.1, {"ratio": 1.0}, penalty=0.4) == pytest.approx(
        0.0
    )
    assert apply_divergence_penalty(0.9, None) == 0.9


def _competition_deps(**overrides: Any) -> SupervisorDeps:
    async def reformulate(_transcript: str) -> EstimationQuery:
        return EstimationQuery(
            function="portal",
            technologies=["python"],
            sector="logistics",
        )

    async def propose_structure(_brief: EstimationQuery) -> AgentStructure:
        return AgentStructure(
            modules=[
                AgentModuleNode(
                    name="Backend",
                    tasks=[AgentTaskNode(name="API", description="REST")],
                )
            ],
            confidence="high",
            reasoning="ok",
        )

    async def backend(args: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": 1,
                "budget_id": "b1",
                "estimated_hours": 40.0,
                "distance": 0.1,
                "content": "api work",
            }
        ]

    async def route(_digest: str) -> SupervisorDecision:
        raise RuntimeError("use fallback ladder")

    async def propose_estimate(stance: str, _brief: str) -> EstimateProposal:
        total = 400.0 if stance == "conservative" else 100.0
        return EstimateProposal(
            stance=stance,  # type: ignore[arg-type]
            total_hours=total,
            assumptions=["a"],
            risks=["r"],
            reasoning=stance,
        )

    async def synthesize(
        proposals: list[dict], divergence: dict
    ) -> SynthesizedEstimate:
        totals = [float(p["total_hours"]) for p in proposals]
        return SynthesizedEstimate(
            low=min(totals),
            high=max(totals),
            driving_assumptions=["spread"],
            open_questions=["scope closed?"],
            confidence="low",
            reasoning="not an average",
        )

    base = dict(
        reformulate=reformulate,
        propose_structure=propose_structure,
        retrieval_backend=backend,
        route_with_model=route,
        confidence_threshold=0.6,
        min_grounded_ratio=0.5,
        out_of_range_factor=2.0,
        max_steps=8,
        privilege_strict=False,
        grounding_max_distance=0.55,
        propose_estimate=propose_estimate,
        synthesize=synthesize,
        competition_enabled=True,
        divergence_penalty=0.4,
    )
    base.update(overrides)
    return SupervisorDeps(**base)


@pytest.mark.asyncio
async def test_both_estimators_run_and_accumulate_proposals():
    deps = _competition_deps()
    graph = build_competition_subgraph(deps)
    result = await graph.ainvoke({"brief": "shared evidence"})
    stances = sorted(p["stance"] for p in result["proposals"])
    assert stances == ["aggressive", "conservative"]


@pytest.mark.asyncio
async def test_synthesizer_returns_a_range_not_an_average():
    deps = _competition_deps()
    graph = build_competition_subgraph(deps)
    result = await graph.ainvoke({"brief": "shared evidence"})
    synthesis = result["synthesis"]
    midpoint = (100 + 400) / 2
    assert synthesis["low"] == 100 and synthesis["high"] == 400
    assert synthesis["low"] != midpoint and synthesis["high"] != midpoint
    assert result["divergence"]["level"] == "high"


def test_subgraph_compiles_with_the_parallel_topology():
    graph = build_competition_subgraph(_competition_deps())
    nodes = set(graph.get_graph().nodes)
    assert {"conservative_estimator", "aggressive_estimator", "synthesizer"} <= nodes


@pytest.mark.asyncio
async def test_competitive_graph_trips_the_gate_on_divergence():
    deps = _competition_deps()
    graph = build_supervisor_graph(deps, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "s14:comp-test"}}
    transcript = (
        "Supplier portal with login, REST API, ERP sync and basic reports for "
        "logistics operations across Spain. Standard enterprise CRUD work. "
    )
    await graph.ainvoke(
        {"transcript": transcript, "estimation_id": "comp-test"},
        config,
    )
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("human_review_gate",)
    values = snapshot.values
    assert values["divergence"]["level"] == "high"
    assert values["estimate"]["range"] == {"low": 100.0, "high": 400.0}


@pytest.mark.asyncio
async def test_competition_failure_returns_base_estimate():
    async def boom_synthesize(proposals, divergence):
        raise RuntimeError("synth failed")

    deps = _competition_deps(synthesize=boom_synthesize)
    nodes = make_supervisor_nodes(deps)
    # Minimal state as if estimate_generator ran after a successful calculate.
    # Drive through the node with a state that has components + matches.
    state = {
        "estimation_id": "e1",
        "supervisor_steps": 3,
        "components": [
            {
                "component_id": "c1",
                "name": "API",
                "category": "Backend",
                "description": "REST",
            }
        ],
        "budget_matches": [
            {
                "component_id": "c1",
                "chunk_id": "1",
                "amount": 40.0,
                "distance": 0.1,
            }
        ],
        "search_completed": True,
    }
    update = await nodes["estimate_generator"](state)
    assert "estimate" in update
    assert "divergence" not in update
    assert any("competition failed" in err for err in (update.get("errors") or []))
