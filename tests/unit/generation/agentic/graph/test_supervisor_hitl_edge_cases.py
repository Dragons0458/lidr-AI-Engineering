"""Session 14 LIVE — HITL edge cases by invariant (not by routing path)."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.domain.supervisor_estimation import (
    SupervisorConflictError,
    resume_supervisor_run,
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
    _ORDER,
)
from app.generation.rag.schemas import EstimationQuery

_EDGE_DIR = (
    Path(__file__).resolve().parents[5] / "exercises" / "session-14" / "edge_cases"
)

_FULL_ROUTE = [
    "requirements_extractor",
    "budget_searcher",
    "estimate_generator",
    "coherence_validator",
    "finish",
]

_SIGNAL_REASON = {
    "low_confidence": "low_confidence",
    "no_precedent": "no_precedent",
    "out_of_historical_range": "out_of_range",
}


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
                tasks=[
                    AgentTaskNode(name="API", description="REST"),
                    AgentTaskNode(name="Auth", description="Login"),
                ],
            )
        ],
        confidence="high",
        reasoning="two tasks",
    )


async def _grounded_backend(_args: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": 1,
            "budget_id": "b1",
            "estimated_hours": 40.0,
            "distance": 0.1,
            "content": "api",
        },
        {
            "id": 2,
            "budget_id": "b2",
            "estimated_hours": 40.0,
            "distance": 0.1,
            "content": "auth",
        },
    ]


async def _empty_backend(_args: Any) -> list[dict[str, Any]]:
    return []


def _scripted_route() -> Any:
    async def _route(_digest: str) -> SupervisorDecision:
        idx = getattr(_route, "_idx", 0)
        target = _FULL_ROUTE[min(idx, len(_FULL_ROUTE) - 1)]
        _route._idx = idx + 1  # type: ignore[attr-defined]
        return SupervisorDecision(
            next_agent=target, reason=f"step {idx}", confidence="high"
        )

    _route._idx = 0  # type: ignore[attr-defined]
    return _route


def _base_deps(**overrides: Any) -> SupervisorDeps:
    defaults = dict(
        reformulate=_reformulate,
        propose_structure=_structure,
        retrieval_backend=_grounded_backend,
        route_with_model=_scripted_route(),
        confidence_threshold=0.6,
        min_grounded_ratio=0.5,
        out_of_range_factor=2.0,
        max_steps=8,
        privilege_strict=False,
        grounding_max_distance=0.45,
    )
    defaults.update(overrides)
    return SupervisorDeps(**defaults)


def _transcript(scenario: str) -> str:
    return (_EDGE_DIR / f"{scenario}.txt").read_text(encoding="utf-8")


async def _run_to_pause(scenario: str, estimation_id: str, deps: SupervisorDeps):
    graph = build_supervisor_graph(deps, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": f"s14:{estimation_id}"}}
    await graph.ainvoke(
        {"transcript": _transcript(scenario), "estimation_id": estimation_id},
        config,
    )
    return graph, config


def _deps_for_signal(scenario: str) -> SupervisorDeps:
    if scenario == "low_confidence":
        return _base_deps(
            confidence_threshold=0.95,
            min_grounded_ratio=0.1,
            out_of_range_factor=50.0,
        )
    if scenario == "no_precedent":
        return _base_deps(
            retrieval_backend=_empty_backend,
            confidence_threshold=0.0,
            min_grounded_ratio=0.5,
            out_of_range_factor=50.0,
        )
    # out_of_historical_range — grounded refs, tight band, inflated total via patch
    return _base_deps(
        confidence_threshold=0.0,
        min_grounded_ratio=0.1,
        out_of_range_factor=2.0,
    )


@contextmanager
def _high_calculate_patch():
    from app.generation.agentic.graph import supervisor_nodes as nodes_mod
    from app.generation.agentic.graph.supervisor_privilege import guarded_dispatch

    real = guarded_dispatch

    async def _patched(agent, tool, args, **kwargs):
        if tool == "calculate_estimate":
            components = args.get("components") or []
            rows = []
            for row in components:
                rows.append(
                    {
                        "name": row.get("name"),
                        "estimated_hours": 400.0,
                        "reference_count": len(row.get("reference_amounts") or []),
                        "unbudgeted": False,
                    }
                )
            return (
                {"ok": True, "total_hours": 800.0, "components": rows},
                {
                    "step": kwargs.get("step", 0),
                    "agent": agent,
                    "action": f"tool:{tool}",
                    "tool": tool,
                    "outcome": "ok",
                    "summary": "stub high total",
                    "args_digest": "stub",
                    "duration_ms": 1,
                },
            )
        return await real(agent, tool, args, **kwargs)

    with patch.object(nodes_mod, "guarded_dispatch", _patched):
        yield


def _scenario_context(scenario: str):
    if scenario == "low_confidence":
        return patch(
            "app.generation.agentic.graph.supervisor_nodes.compute_confidence",
            return_value=0.2,
        )
    if scenario == "out_of_historical_range":
        return _high_calculate_patch()
    return nullcontext()


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", list(_SIGNAL_REASON))
async def test_each_signal_trips_the_pause(scenario):
    deps = _deps_for_signal(scenario)
    with _scenario_context(scenario):
        graph, config = await _run_to_pause(scenario, f"edge-{scenario}", deps)

    snap = await graph.aget_state(config)
    assert snap.next == ("human_review_gate",)
    payload = snap.interrupts[0].value
    assert payload["gate"] == "low_confidence_review"
    assert _SIGNAL_REASON[scenario] in payload["reasons"]


@pytest.mark.asyncio
async def test_paused_state_persisted_in_checkpoint():
    with patch(
        "app.generation.agentic.graph.supervisor_nodes.compute_confidence",
        return_value=0.2,
    ):
        graph, config = await _run_to_pause(
            "low_confidence", "edge-persist", _deps_for_signal("low_confidence")
        )
    snap = await graph.aget_state(config)
    assert snap.next == ("human_review_gate",)
    assert snap.values.get("estimate") is not None
    assert snap.values.get("confidence") is not None


@pytest.mark.asyncio
async def test_resume_continues_with_human_decision():
    with patch(
        "app.generation.agentic.graph.supervisor_nodes.compute_confidence",
        return_value=0.2,
    ):
        graph, config = await _run_to_pause(
            "low_confidence", "edge-resume", _deps_for_signal("low_confidence")
        )
    await graph.ainvoke(
        Command(resume={"decision": "approve", "note": "checked"}), config
    )
    final = await graph.aget_state(config)
    assert final.next == ()
    assert final.values["human_decision"]["decision"] == "approve"
    assert final.values["status"] == "validated"


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", list(_SIGNAL_REASON))
async def test_no_agent_before_preconditions(scenario):
    deps = _deps_for_signal(scenario)
    with _scenario_context(scenario):
        graph, config = await _run_to_pause(scenario, f"edge-precond-{scenario}", deps)
    snap = await graph.aget_state(config)
    dispatched = [
        row["next_agent"]
        for row in snap.values.get("routing_history") or []
        if row["next_agent"] in _ORDER
    ]
    positions = [_ORDER.index(name) for name in dispatched]
    assert positions == sorted(positions)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", list(_SIGNAL_REASON))
async def test_step_budget_not_exceeded(scenario):
    deps = _deps_for_signal(scenario)
    with _scenario_context(scenario):
        graph, config = await _run_to_pause(scenario, f"edge-budget-{scenario}", deps)
    snap = await graph.aget_state(config)
    assert snap.values.get("supervisor_steps", 0) <= deps.max_steps


@pytest.mark.asyncio
async def test_second_resume_raises_supervisor_conflict():
    with patch(
        "app.generation.agentic.graph.supervisor_nodes.compute_confidence",
        return_value=0.2,
    ):
        graph, config = await _run_to_pause(
            "low_confidence", "edge-idem", _deps_for_signal("low_confidence")
        )
    await graph.ainvoke(Command(resume={"decision": "approve"}), config)
    runtime = SimpleNamespace(supervisor_graph=graph)
    with pytest.raises(SupervisorConflictError):
        await resume_supervisor_run("edge-idem", {"decision": "reject"}, runtime)
