"""Multi-agent nodes for the Session 13 live LangGraph flow.

External I/O is injected via ``MultiAgentDeps`` so unit tests swap fakes without
touching singletons. Personas are resolved in wiring, not inside nodes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import logfire
import structlog
from langgraph.types import Command, interrupt

from app.generation.agentic.agent_schemas import (
    AgentStructure,
    AgentTaskHoursRun,
    AgentTaskRef,
)
from app.generation.agentic.graph.estimate_builder import (
    build_estimate,
    flag_reason,
    modules_from_structure,
    recompute_estimate_totals,
)
from app.generation.agentic.graph.nodes import _estimation_id
from app.generation.agentic.graph.schemas import (
    CommercialProposal,
    ComplexityClassification,
    ReliabilityReport,
)
from app.generation.agentic.graph.state import EstimationState
from app.generation.rag.schemas import TaskHoursEstimate

log = structlog.get_logger()


@dataclass(frozen=True)
class MultiAgentDeps:
    """Injectable collaborators for the eight multi-agent nodes."""

    classify: Callable[[str], Awaitable[ComplexityClassification]]
    propose_structure: Callable[[str, str], Awaitable[AgentStructure]]
    estimate_task: Callable[[str, str, str | None], Awaitable[TaskHoursEstimate]]
    recover: Callable[[list[AgentTaskRef]], Awaitable[AgentTaskHoursRun]] | None
    analyze: Callable[[str], Awaitable[ReliabilityReport]]
    propose: Callable[[str], Awaitable[CommercialProposal]]
    recovery_reliability_threshold: float
    structure_effort_by_complexity: dict[str, str]
    default_reasoning_effort: str


def _grounded_ratio(estimate: dict) -> float:
    tasks = [t for m in estimate.get("modules") or [] for t in (m.get("tasks") or [])]
    if not tasks:
        return 0.0
    grounded = sum(1 for t in tasks if t.get("estimated_hours") is not None)
    return round(grounded / len(tasks), 3)


def _estimate_digest(estimate: dict, ratio: float) -> str:
    lines = [
        f"total_engineer_days: {estimate.get('total_engineer_days')}",
        f"total_engineer_hours: {estimate.get('total_engineer_hours')}",
        f"grounded_task_ratio: {ratio}",
        "tasks:",
    ]
    for module in estimate.get("modules") or []:
        for task in module.get("tasks") or []:
            hours = task.get("estimated_hours")
            hours_text = f"{hours}h" if hours is not None else "NO MATCH"
            lines.append(
                f"  - [{module.get('name')}] {task.get('name')}: {hours_text} "
                f"(reliability={task.get('reliability')}, has_match={task.get('has_match')})"
            )
    return "\n".join(lines)


def make_multiagent_nodes(deps: MultiAgentDeps) -> dict[str, Callable[..., Any]]:
    """Build the eight node callables closed over the injected dependencies."""

    async def classifier_agent(state: EstimationState) -> Command:
        estimation_id = _estimation_id()
        with logfire.span("agent.graph.classifier_agent", estimation_id=estimation_id):
            result = await deps.classify(state["transcript"])
            log.info(
                "graph_classifier_agent",
                complexity=result.complexity,
                brief_chars=len(result.reformulated_transcript),
                estimation_id=estimation_id,
            )
            return Command(
                goto="structure_agent",
                update={
                    "complexity": result.complexity,
                    "reformulated_transcript": result.reformulated_transcript,
                },
            )

    async def structure_agent(state: EstimationState) -> dict[str, Any]:
        estimation_id = _estimation_id()
        with logfire.span("agent.graph.structure_agent", estimation_id=estimation_id):
            brief = (
                state.get("reformulated_transcript") or state.get("transcript") or ""
            )
            complexity = state.get("complexity") or "medium"
            effort = deps.structure_effort_by_complexity.get(
                complexity, deps.default_reasoning_effort
            )
            structure = await deps.propose_structure(brief, effort)
            task_count = sum(len(m.tasks) for m in structure.modules)
            log.info(
                "graph_structure_agent",
                modules=len(structure.modules),
                tasks=task_count,
                effort=effort,
                estimation_id=estimation_id,
            )
            return {"structure": structure.model_dump(mode="json")}

    async def human_gate_structure(state: EstimationState) -> dict[str, Any]:
        decision = interrupt(
            {
                "gate": "structure_review",
                "estimation_id": state.get("estimation_id"),
                "complexity": state.get("complexity"),
                "structure": state.get("structure"),
            }
        )
        estimation_id = _estimation_id()
        with logfire.span(
            "agent.graph.human_gate_structure", estimation_id=estimation_id
        ):
            decision = decision or {}
            modules = decision.get("modules") or modules_from_structure(
                state.get("structure")
            )
            log.info(
                "graph_human_gate_structure",
                approved=decision.get("approved"),
                modules=len(modules),
                estimation_id=estimation_id,
            )
            return {"approved_modules": modules, "gate1_decision": decision}

    async def estimate_task_hours(state: EstimationState) -> dict[str, Any]:
        estimation_id = _estimation_id()
        with logfire.span(
            "agent.graph.estimate_task_hours", estimation_id=estimation_id
        ):
            module = state["module"]
            task = state["task"]
            description = state.get("description")
            est = await deps.estimate_task(module, task, description)
            log.info(
                "graph_estimate_task_hours",
                module=module,
                task=task,
                has_match=est.has_match,
                hours=est.estimated_hours,
                estimation_id=estimation_id,
            )
            return {"task_hours": [est.model_dump(mode="json")]}

    async def recover_and_handover(state: EstimationState) -> Command:
        estimation_id = _estimation_id()
        with logfire.span(
            "agent.graph.recover_and_handover", estimation_id=estimation_id
        ):
            approved = state.get("approved_modules") or []
            task_hours = list(state.get("task_hours") or [])
            by_key = {(t.get("module"), t.get("task")): t for t in task_hours}
            descriptions = {
                (m.get("name"), t.get("name")): t.get("description")
                for m in approved
                for t in (m.get("tasks") or [])
            }

            flagged: list[AgentTaskRef] = []
            flagged_map: dict[str, tuple[str, str]] = {}
            for index, row in enumerate(task_hours):
                reason = flag_reason(
                    row, reliability_threshold=deps.recovery_reliability_threshold
                )
                if reason is None:
                    continue
                task_ref = f"task-{index}"
                flagged.append(
                    AgentTaskRef(
                        task_ref=task_ref,
                        module=row.get("module") or "",
                        task=row.get("task") or "",
                        description=descriptions.get(
                            (row.get("module"), row.get("task"))
                        ),
                        reason=reason,
                    )
                )
                flagged_map[task_ref] = (row.get("module") or "", row.get("task") or "")

            merged = task_hours
            recovered_count = 0
            errors: list[str] = []
            if flagged and deps.recover is not None:
                log.info(
                    "graph_agentic_recovery_start",
                    flagged=len(flagged),
                    total=len(task_hours),
                    estimation_id=estimation_id,
                )
                run = await deps.recover(flagged)
                recovered: dict[tuple[str, str], Any] = {}
                for derivation in run.derivations:
                    identity = flagged_map.get(derivation.task_ref)
                    if identity is None:
                        errors.append(
                            f"Recovery returned unknown task_ref: {derivation.task_ref}"
                        )
                        continue
                    module_name, task_name = identity
                    if derivation.module != module_name or derivation.task != task_name:
                        errors.append(
                            f"Recovery altered task identity for {derivation.task_ref}"
                        )
                        continue
                    if not derivation.has_match or derivation.estimated_hours is None:
                        continue
                    recovered[(module_name, task_name)] = derivation
                recovered_count = len(recovered)
                merged_map = dict(by_key)
                for key, derivation in recovered.items():
                    base = merged_map.get(key, {"module": key[0], "task": key[1]})
                    merged_map[key] = {
                        **base,
                        "estimated_hours": derivation.estimated_hours,
                        "reliability": derivation.reliability,
                        "has_match": True,
                        "hours_range": None,
                        "estimation_source": "agent_recovery",
                    }
                merged = list(merged_map.values())

            estimate = build_estimate(approved, merged)
            log.info(
                "graph_recover_and_handover",
                flagged=len(flagged),
                recovered=recovered_count,
                total_engineer_days=estimate.get("total_engineer_days"),
                estimation_id=estimation_id,
            )
            update: dict[str, Any] = {"estimate": estimate, "task_hours": merged}
            if errors:
                update["errors"] = errors
            return Command(goto="analysis_agent", update=update)

    async def analysis_agent(state: EstimationState) -> dict[str, Any]:
        estimation_id = _estimation_id()
        with logfire.span("agent.graph.analysis_agent", estimation_id=estimation_id):
            estimate = state.get("estimate") or {}
            ratio = _grounded_ratio(estimate)
            digest = _estimate_digest(estimate, ratio)
            report = await deps.analyze(digest)
            report.grounded_task_ratio = ratio
            log.info(
                "graph_analysis_agent",
                overall_confidence=report.overall_confidence,
                grounded_task_ratio=ratio,
                weak_points=len(report.weak_points),
                estimation_id=estimation_id,
            )
            return {"analysis_report": report.model_dump(mode="json")}

    async def human_gate_analysis(state: EstimationState) -> dict[str, Any]:
        decision = interrupt(
            {
                "gate": "final_review",
                "estimation_id": state.get("estimation_id"),
                "estimate": state.get("estimate"),
                "analysis_report": state.get("analysis_report"),
            }
        )
        estimation_id = _estimation_id()
        with logfire.span(
            "agent.graph.human_gate_analysis", estimation_id=estimation_id
        ):
            decision = decision or {}
            overrides = decision.get("estimate_overrides") or {}
            estimate = {**(state.get("estimate") or {}), **overrides}
            if estimate.get("modules"):
                estimate = {
                    **estimate,
                    **recompute_estimate_totals(estimate["modules"]),
                }
            status = "validated" if decision.get("validated") else "needs_review"
            log.info(
                "graph_human_gate_analysis",
                validated=decision.get("validated"),
                want_proposal=decision.get("want_proposal"),
                overrides=len(overrides),
                estimation_id=estimation_id,
            )
            return {
                "estimate": estimate,
                "gate2_decision": decision,
                "status": status,
            }

    async def proposal_agent(state: EstimationState) -> dict[str, Any]:
        estimation_id = _estimation_id()
        with logfire.span("agent.graph.proposal_agent", estimation_id=estimation_id):
            estimate = state.get("estimate") or {}
            analysis_report = state.get("analysis_report") or {}
            lines = [
                f"total_engineer_days: {estimate.get('total_engineer_days')}",
                f"confidence: {estimate.get('confidence')}",
                f"reliability_summary: {analysis_report.get('summary', '')}",
                "modules:",
            ]
            for module in estimate.get("modules") or []:
                task_hours = [
                    t.get("estimated_hours")
                    for t in (module.get("tasks") or [])
                    if t.get("estimated_hours")
                ]
                lines.append(
                    f"  - {module.get('name')}: {len(module.get('tasks') or [])} tasks, "
                    f"{sum(task_hours)}h total"
                )
            proposal = await deps.propose("\n".join(lines))
            log.info(
                "graph_proposal_agent",
                title=proposal.title,
                scope=len(proposal.scope),
                estimation_id=estimation_id,
            )
            return {"proposal": proposal.body_markdown}

    return {
        "classifier_agent": classifier_agent,
        "structure_agent": structure_agent,
        "human_gate_structure": human_gate_structure,
        "estimate_task_hours": estimate_task_hours,
        "recover_and_handover": recover_and_handover,
        "analysis_agent": analysis_agent,
        "human_gate_analysis": human_gate_analysis,
        "proposal_agent": proposal_agent,
    }


__all__ = ["MultiAgentDeps", "make_multiagent_nodes"]
