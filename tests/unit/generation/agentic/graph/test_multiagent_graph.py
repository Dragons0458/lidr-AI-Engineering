"""End-to-end multi-agent graph run, network-free."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.generation.agentic.agent_schemas import (
    AgentModuleNode,
    AgentStructure,
    AgentTaskDerivation,
    AgentTaskHoursRun,
    AgentTaskNode,
)
from app.generation.agentic.graph.agent_nodes import MultiAgentDeps
from app.generation.agentic.graph.build import (
    build_estimation_graph,
    fan_out_hours,
    route_after_gate2,
)
from app.generation.agentic.graph.schemas import (
    CommercialProposal,
    ComplexityClassification,
    ReliabilityReport,
)
from app.domain.schemas.agent_trace import AgentTrace
from app.generation.rag.schemas import TaskHoursEstimate

TRANSCRIPT = "A" * 200
CONFIG = {"configurable": {"thread_id": "t1"}}


def _fake_deps(
    *,
    complexity="high",
    structure_modules=None,
    hours_by_task=None,
    no_match=(),
    recovery_fn=None,
):
    structure_modules = structure_modules or {"Backend": ["API", "Auth"]}
    hours_by_task = hours_by_task or {"API": 80, "Auth": 40}

    async def classify(transcript: str) -> ComplexityClassification:
        return ComplexityClassification(
            complexity=complexity,
            reformulated_transcript="Build a backend and a mobile app.",
            reasoning="several components",
        )

    async def propose_structure(brief: str, reasoning_effort: str) -> AgentStructure:
        return AgentStructure(
            modules=[
                AgentModuleNode(
                    name=module,
                    tasks=[
                        AgentTaskNode(name=task, description=f"{task} scope")
                        for task in tasks
                    ],
                )
                for module, tasks in structure_modules.items()
            ],
            confidence="high",
            reasoning="decomposed",
        )

    async def estimate_task(module, name, description):
        if name in no_match:
            return TaskHoursEstimate(module=module, task=name, has_match=False)
        return TaskHoursEstimate(
            module=module,
            task=name,
            estimated_hours=hours_by_task.get(name, 40),
            reliability=0.85,
            has_match=True,
            dispersion=0.1,
            neighbors=[],
        )

    async def analyze(digest: str) -> ReliabilityReport:
        return ReliabilityReport(
            overall_confidence="medium",
            grounded_task_ratio=1.0,
            weak_points=[],
            summary="looks reasonable",
        )

    async def propose(user_message: str) -> CommercialProposal:
        return CommercialProposal(
            title="RUTA",
            executive_summary="A logistics platform.",
            scope=["Backend"],
            total_engineer_days=20,
            body_markdown="# Proposal\n...",
        )

    return MultiAgentDeps(
        classify=classify,
        propose_structure=propose_structure,
        estimate_task=estimate_task,
        recover=recovery_fn,
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


async def _start(graph):
    return await graph.ainvoke(
        {"transcript": TRANSCRIPT, "estimation_id": "t1"},
        CONFIG,
    )


@pytest.mark.asyncio
async def test_full_flow_pauses_at_both_gates_and_completes():
    deps = _fake_deps(
        structure_modules={"Backend": ["API", "Auth"], "Mobile": ["App"]},
        hours_by_task={"API": 80, "Auth": 40, "App": 120},
    )
    graph = build_estimation_graph(deps, checkpointer=MemorySaver())

    await _start(graph)
    snap = await graph.aget_state(CONFIG)
    assert snap.next == ("human_gate_structure",)
    assert snap.interrupts[0].value["gate"] == "structure_review"

    await graph.ainvoke(Command(resume={"approved": True}), CONFIG)
    snap = await graph.aget_state(CONFIG)
    assert snap.next == ("human_gate_analysis",)
    assert len(snap.values["task_hours"]) == 3

    result = await graph.ainvoke(
        Command(resume={"validated": True, "want_proposal": True}),
        CONFIG,
    )
    snap = await graph.aget_state(CONFIG)
    assert snap.next == ()
    assert result["status"] == "validated"
    assert result["proposal"].startswith("# Proposal")


@pytest.mark.asyncio
async def test_gate2_without_proposal_ends_without_proposal():
    deps = _fake_deps(structure_modules={"Backend": ["API"]}, hours_by_task={"API": 40})
    graph = build_estimation_graph(
        deps, checkpointer=MemorySaver(), proposal_enabled=True
    )
    await _start(graph)
    await graph.ainvoke(Command(resume={"approved": True}), CONFIG)
    result = await graph.ainvoke(
        Command(resume={"validated": True, "want_proposal": False}),
        CONFIG,
    )
    assert result.get("proposal") is None


@pytest.mark.asyncio
async def test_flagged_task_triggers_recovery_once():
    recovery_calls: list[int] = []

    async def recovery(flagged, **kwargs):
        recovery_calls.append(len(flagged))
        return AgentTaskHoursRun(
            derivations=[
                AgentTaskDerivation(
                    task_ref=item.task_ref,
                    module=item.module,
                    task=item.task,
                    estimated_hours=64,
                    reliability=0.7,
                    has_match=True,
                    neighbors=[
                        {
                            "source_id": 1,
                            "budget_id": "B1",
                            "estimated_hours": 64,
                            "distance": 0.2,
                        }
                    ],
                )
                for item in flagged
            ],
            trace=AgentTrace(),
            iterations=1,
            stopped_reason="completed",
        )

    deps = _fake_deps(
        structure_modules={"Backend": ["API", "Legacy"]},
        hours_by_task={"API": 40},
        no_match={"Legacy"},
        recovery_fn=recovery,
    )
    graph = build_estimation_graph(deps, checkpointer=MemorySaver())
    await _start(graph)
    await graph.ainvoke(Command(resume={"approved": True}), CONFIG)
    snap = await graph.aget_state(CONFIG)
    assert recovery_calls == [1]
    hours = {row["task"]: row["estimated_hours"] for row in snap.values["task_hours"]}
    assert hours == {"API": 40, "Legacy": 64}


def test_fan_out_hours_emits_one_send_per_task():
    state = {
        "approved_modules": [
            {"name": "Backend", "tasks": [{"name": "API"}, {"name": "Auth"}]},
            {"name": "Mobile", "tasks": [{"name": "App"}]},
        ]
    }
    sends = fan_out_hours(state)
    assert [send.arg["task"] for send in sends] == ["API", "Auth", "App"]


def test_fan_out_hours_with_no_tasks_routes_to_join():
    assert fan_out_hours({"approved_modules": []}) == "recover_and_handover"


def test_route_after_gate2_honours_want_proposal():
    assert route_after_gate2({"gate2_decision": {"want_proposal": True}}) == "proposal"
    assert route_after_gate2({"gate2_decision": {"want_proposal": False}}) == "end"
    assert route_after_gate2({}) == "end"
