"""Domain conductor for the Session 12 hybrid agent workflow."""

from __future__ import annotations

from typing import Any

import structlog

from app.generation.agentic.agent_loop import (
    run_structure_agent,
    run_task_hours_recovery_agent,
)
from app.generation.agentic.agent_schemas import AgentTaskRef
from app.generation.rag.agent_retrieval import make_retrieval_backend
from app.generation.rag.prompt_builder import build_structure_user_message
from app.generation.rag.schemas import (
    Estimate,
    EstimationQuery,
    GenerateResult,
    TaskHoursEstimate,
    TaskHoursModuleInput,
    TaskHoursResult,
    TaskItem,
    TaskNeighbor,
    WorkModule,
)
from app.generation.rag.task_hours import (
    distance_weighted_consensus,
    estimate_all,
)

log = structlog.get_logger()


class OpenAIClientMissingError(RuntimeError):
    """Raised when recovery is required but no OpenAI client is configured."""


class RecoveryAgentError(RuntimeError):
    """Raised when the recovery loop fails or violates conductor identity rules."""


async def agent_propose_structure(
    query: EstimationQuery,
    *,
    client: Any,
    model: str,
    reasoning_effort: str,
    persona: str | None,
) -> GenerateResult:
    """Create a tool-free structure compatible with the existing wizard."""
    brief = build_structure_user_message(query)
    structure, trace = await run_structure_agent(
        brief,
        client=client,
        model=model,
        reasoning_effort=reasoning_effort,
        persona=persona,
    )
    modules = [
        WorkModule(
            name=module.name,
            description=module.description,
            tasks=[
                TaskItem(
                    name=task.name,
                    description=task.description,
                    engineer_days=None,
                    grounded=False,
                    sources=[],
                )
                for task in module.tasks
            ],
        )
        for module in structure.modules
    ]
    confidence = structure.confidence if modules else "insufficient"
    estimate = Estimate(
        total_engineer_days=None,
        modules=modules,
        duration_weeks=None,
        sources=[],
        assumptions=[],
        confidence=confidence,
        reasoning=structure.reasoning,
        insufficient_context_explanation=structure.insufficient_context_explanation,
    )
    return GenerateResult(
        estimate=estimate,
        fabricated_source_ids=[],
        coherent=True,
        agent_trace=trace,
    )


def _flag_reason(task: TaskHoursEstimate, threshold: float) -> str | None:
    reasons: list[str] = []
    if not task.has_match:
        reasons.append("no historical match")
    if task.hours_range is not None:
        reasons.append("contradictory historical range")
    if task.reliability is not None and task.reliability < threshold:
        reasons.append(f"reliability below {threshold}")
    return "; ".join(reasons) or None


async def agent_estimate_task_hours(
    modules: list[TaskHoursModuleInput],
    *,
    client: Any | None,
    model: str,
    reasoning_effort: str,
    max_iterations: int,
    top_k: int,
    distance_threshold: float,
    search_mode: str,
    rerank: bool,
    persona: str | None,
    recovery_reliability_threshold: float,
) -> TaskHoursResult:
    """Run deterministic estimation, then selectively recover flagged tasks."""
    deterministic = await estimate_all(
        modules,
        top_k=top_k,
        distance_threshold=distance_threshold,
    )
    flat_inputs = [(module, task) for module in modules for task in module.tasks]
    flagged: list[AgentTaskRef] = []
    flagged_map: dict[str, tuple[int, str, str]] = {}
    for index, (result, (module, task)) in enumerate(
        zip(deterministic.tasks, flat_inputs, strict=True)
    ):
        reason = _flag_reason(result, recovery_reliability_threshold)
        if reason is None:
            continue
        task_ref = f"task-{index}"
        flagged.append(
            AgentTaskRef(
                task_ref=task_ref,
                module=module.name,
                task=task.name,
                description=task.description,
                reason=reason,
            )
        )
        flagged_map[task_ref] = (index, module.name, task.name)

    log.info(
        "agent_hours_flags",
        tasks=len(deterministic.tasks),
        flagged=len(flagged),
    )
    if not flagged:
        from app.domain.schemas.agent_trace import AgentTrace

        return TaskHoursResult(tasks=deterministic.tasks, agent_trace=AgentTrace())
    if client is None:
        raise OpenAIClientMissingError(
            "OpenAI client is required when task-hours recovery is needed."
        )

    backend = make_retrieval_backend(
        top_k=top_k,
        distance_threshold=distance_threshold,
        search_mode=search_mode,
        rerank=rerank,
    )
    try:
        run = await run_task_hours_recovery_agent(
            flagged,
            client=client,
            model=model,
            reasoning_effort=reasoning_effort,
            max_iterations=max_iterations,
            backend=backend,
            consensus=distance_weighted_consensus,
            persona=persona,
        )
        merged = list(deterministic.tasks)
        for derivation in run.derivations:
            identity = flagged_map.get(derivation.task_ref)
            if identity is None:
                raise RecoveryAgentError("Recovery returned an unknown task_ref.")
            index, module_name, task_name = identity
            if derivation.module != module_name or derivation.task != task_name:
                raise RecoveryAgentError("Recovery altered the task identity.")
            if not derivation.has_match or derivation.estimated_hours is None:
                continue
            merged[index] = TaskHoursEstimate(
                module=module_name,
                task=task_name,
                estimated_hours=derivation.estimated_hours,
                reliability=derivation.reliability,
                dispersion=derivation.dispersion,
                has_match=True,
                neighbors=[
                    TaskNeighbor(
                        source_id=neighbor.source_id,
                        budget_id=neighbor.budget_id,
                        estimated_hours=neighbor.estimated_hours,
                        distance=neighbor.distance,
                    )
                    for neighbor in derivation.neighbors
                ],
                hours_range=None,
                estimation_source="agent_recovery",
            )
        return TaskHoursResult(tasks=merged, agent_trace=run.trace)
    except RecoveryAgentError:
        raise
    except Exception as exc:
        raise RecoveryAgentError("Task-hours recovery failed.") from exc
