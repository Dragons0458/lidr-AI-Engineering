"""Domain conductor for the Session 13 LangGraph multi-agent estimation workflow."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from langgraph.types import Command

from app.domain.schemas.graph_estimation import (
    ActivityEntry,
    GraphProgress,
    GraphRunState,
    PendingGate,
)
from app.foundation.persistence.langgraph import LangGraphRuntime
from app.generation.agentic.agent_loop import (
    run_structure_agent,
    run_task_hours_recovery_agent,
)
from app.generation.agentic.agent_schemas import AgentStructure, AgentTaskRef
from app.generation.agentic.graph.activity import GraphActivityLog, describe_node
from app.generation.agentic.graph.agent_nodes import MultiAgentDeps
from app.generation.agentic.graph.build import build_estimation_graph
from app.generation.agentic.graph.nodes import GraphNodeDeps
from app.generation.agentic.graph.personas import persona_for
from app.generation.agentic.graph.schemas import (
    CommercialProposal,
    ComplexityClassification,
    ReliabilityReport,
)
from app.generation.rag.agent_retrieval import make_retrieval_backend
from app.generation.rag.prompt_builder import build_structure_user_message
from app.generation.rag.query_reformulator import reformulate_query
from app.generation.rag.schemas import EstimationQuery, TaskHoursEstimate
from app.generation.rag.task_hours import distance_weighted_consensus, estimate_one

log = structlog.get_logger()

_CLASSIFIER_SYSTEM_PROMPT = (
    "You are an estimation triage analyst. You are given a raw, messy client meeting "
    "transcript (any language). Do TWO things:\n"
    "1. Judge the COMPLEXITY of the estimation this project will require: 'low' (a "
    "single simple component), 'medium' (a few related components) or 'high' (many "
    "disparate components and/or third-party integrations).\n"
    "2. REFORMULATE the transcript into a clean, self-contained project brief in "
    "concise technical English: the components the client wants, their scope and "
    "constraints, with small talk, anecdotes and digressions removed. Never invent "
    "scope the transcript gives no evidence for.\n"
    "Return the complexity, the reformulated brief and one line on why."
)

_ANALYSIS_SYSTEM_PROMPT = (
    "You are an estimation reviewer. You are given a structured software estimate: "
    "modules → tasks, each task with derived engineer-hours, a reliability score "
    "(0..1) and whether it matched a historical analog. Write a RELIABILITY REPORT for "
    "the human who will approve it:\n"
    "- overall_confidence: how much to trust the estimate as a whole.\n"
    "- grounded_task_ratio: the fraction of tasks with grounded hours (use the value "
    "given in the input; do not recompute).\n"
    "- weak_points: the specific soft spots the human must check or complete — tasks "
    "with no match, low reliability, or contradictory analogs. Be concrete.\n"
    "- summary: a short honest prose read. Never invent numbers; only judge the ones given."
)

_PROPOSAL_SYSTEM_PROMPT = (
    "You are a delivery lead writing a concise commercial proposal for a client, based "
    "STRICTLY on a validated software estimate (modules → tasks with engineer-days) and "
    "its reliability report. Write a title, a 2-4 sentence executive summary, a bullet "
    "scope of the modules/deliverables, echo the total engineer-days, and a full "
    "proposal body in Markdown. Do NOT invent scope, prices or numbers not present in "
    "the estimate. Keep it honest and client-ready."
)


class GraphRuntimeUnavailableError(RuntimeError):
    """Raised when the LangGraph runtime was not started (e.g. no Postgres)."""


class GraphEstimationError(RuntimeError):
    """Raised when the graph run fails for a technical reason."""


class GraphConflictError(RuntimeError):
    """Raised when a resume or proposal is requested but the run state forbids it."""


class GraphNotFoundError(RuntimeError):
    """Raised when no checkpoint exists for the given estimation_id."""


def _require_runtime(runtime: LangGraphRuntime | None) -> LangGraphRuntime:
    if runtime is None:
        raise GraphRuntimeUnavailableError(
            "LangGraph estimation runtime is not available."
        )
    return runtime


def snapshot_to_run_state(estimation_id: str, snapshot: Any) -> GraphRunState:
    """Turn a LangGraph ``StateSnapshot`` into the public ``GraphRunState``."""
    values = snapshot.values or {}
    paused = bool(snapshot.next)
    pending_gate = None
    interrupts = getattr(snapshot, "interrupts", None) or ()
    if paused and interrupts:
        gate_value = interrupts[0].value or {}
        pending_gate = PendingGate(
            gate=gate_value.get("gate", "unknown"),
            estimation_id=estimation_id,
            payload={
                key: value
                for key, value in gate_value.items()
                if key not in ("gate", "estimation_id")
            },
        )
    return GraphRunState(
        estimation_id=estimation_id,
        state="paused" if paused else "completed",
        pending_gate=pending_gate,
        complexity=values.get("complexity"),
        structure=values.get("structure"),
        task_hours=values.get("task_hours") or [],
        estimate=values.get("estimate"),
        analysis_report=values.get("analysis_report"),
        proposal=values.get("proposal"),
        status=values.get("status"),
        errors=values.get("errors") or [],
    )


def progress_state(snapshot: Any) -> str:
    """``running`` (mid-leg) | ``paused`` (at a gate) | ``completed`` (END)."""
    if not getattr(snapshot, "next", None):
        return "completed"
    interrupts = getattr(snapshot, "interrupts", None) or ()
    return "paused" if interrupts else "running"


def _proposal_input(estimate: dict, analysis_report: dict) -> str:
    lines = [
        f"total_engineer_days: {estimate.get('total_engineer_days')}",
        f"confidence: {estimate.get('confidence')}",
        f"reliability_summary: {(analysis_report or {}).get('summary', '')}",
        "modules:",
    ]
    for module in estimate.get("modules") or []:
        task_hours = [
            task.get("estimated_hours")
            for task in (module.get("tasks") or [])
            if task.get("estimated_hours")
        ]
        lines.append(
            f"  - {module.get('name')}: {len(module.get('tasks') or [])} tasks, "
            f"{sum(task_hours)}h total"
        )
    return "\n".join(lines)


def build_default_graph_deps(
    *,
    client: Any,
    model: str,
    reasoning_effort: str,
    top_k: int,
    distance_threshold: float,
    search_mode: str,
    rerank: bool,
) -> GraphNodeDeps:
    """Wire production adapters for the legacy sequential graph (tests)."""

    async def reformulate(transcript: str) -> EstimationQuery:
        return await reformulate_query(transcript)

    async def propose_structure(brief: EstimationQuery) -> AgentStructure:
        user_message = build_structure_user_message(brief)
        structure, _trace = await run_structure_agent(
            user_message,
            client=client,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        return structure

    recovery = make_retrieval_backend(
        top_k=top_k,
        distance_threshold=distance_threshold,
        search_mode=search_mode,
        rerank=rerank,
    )
    from app.generation.agentic.agent_tools import adapt_recovery_backend

    return GraphNodeDeps(
        reformulate=reformulate,
        propose_structure=propose_structure,
        retrieval_backend=adapt_recovery_backend(recovery),
    )


def build_default_multiagent_deps(
    *,
    client: Any,
    settings: Any,
    runtime_retrieval: Any,
) -> MultiAgentDeps:
    """Wire production adapters for the multi-agent graph."""
    from app.dependencies import get_llm_wrapper

    wrapper = get_llm_wrapper()
    personas_enabled = settings.GRAPH_PERSONAS_ENABLED
    top_k = runtime_retrieval.effective_task_hours_top_k()
    distance_threshold = runtime_retrieval.effective_task_hours_distance_threshold()
    search_mode = runtime_retrieval.effective_search_mode()
    rerank = runtime_retrieval.effective_rerank()

    async def classify(transcript: str) -> ComplexityClassification:
        persona = persona_for("classifier_agent", enabled=personas_enabled)
        system_prompt = (
            f"{persona}\n\n{_CLASSIFIER_SYSTEM_PROMPT}"
            if persona
            else _CLASSIFIER_SYSTEM_PROMPT
        )
        result, _meta = await asyncio.to_thread(
            wrapper.complete_structured,
            system_prompt=system_prompt,
            user_message=transcript,
            response_model=ComplexityClassification,
            model_override=settings.GRAPH_CLASSIFIER_MODEL,
        )
        return result

    async def propose_structure(brief: str, reasoning_effort: str) -> AgentStructure:
        persona = persona_for("structure_agent", enabled=personas_enabled)
        structure, _trace = await run_structure_agent(
            brief,
            client=client,
            model=settings.AGENT_MODEL,
            reasoning_effort=reasoning_effort,
            persona=persona,
        )
        return structure

    async def estimate_task(
        module: str, task: str, description: str | None
    ) -> TaskHoursEstimate:
        return await estimate_one(
            module,
            task,
            description,
            top_k=top_k,
            distance_threshold=distance_threshold,
            search_mode=search_mode,
            rerank=rerank,
        )

    async def recover(flagged: list[AgentTaskRef]):
        if client is None:
            raise GraphEstimationError(
                "OpenAI client required for task-hours recovery."
            )
        backend = make_retrieval_backend(
            top_k=top_k,
            distance_threshold=distance_threshold,
            search_mode=search_mode,
            rerank=rerank,
        )
        persona = persona_for("recover_and_handover", enabled=personas_enabled)
        return await run_task_hours_recovery_agent(
            flagged,
            client=client,
            model=settings.AGENT_MODEL,
            reasoning_effort=settings.AGENT_REASONING_EFFORT,
            max_iterations=settings.AGENT_MAX_ITERATIONS,
            backend=backend,
            consensus=distance_weighted_consensus,
            persona=persona,
        )

    async def analyze(digest: str) -> ReliabilityReport:
        persona = persona_for("analysis_agent", enabled=personas_enabled)
        system_prompt = (
            f"{persona}\n\n{_ANALYSIS_SYSTEM_PROMPT}"
            if persona
            else _ANALYSIS_SYSTEM_PROMPT
        )
        report, _meta = await asyncio.to_thread(
            wrapper.complete_structured,
            system_prompt=system_prompt,
            user_message=digest,
            response_model=ReliabilityReport,
            model_override=settings.GRAPH_ANALYSIS_MODEL,
        )
        return report

    async def propose(user_message: str) -> CommercialProposal:
        persona = persona_for("proposal_agent", enabled=personas_enabled)
        system_prompt = (
            f"{persona}\n\n{_PROPOSAL_SYSTEM_PROMPT}"
            if persona
            else _PROPOSAL_SYSTEM_PROMPT
        )
        proposal, _meta = await asyncio.to_thread(
            wrapper.complete_structured,
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=CommercialProposal,
            model_override=settings.GRAPH_PROPOSAL_MODEL,
        )
        return proposal

    return MultiAgentDeps(
        classify=classify,
        propose_structure=propose_structure,
        estimate_task=estimate_task,
        recover=recover if client is not None else None,
        analyze=analyze,
        propose=propose,
        recovery_reliability_threshold=settings.AGENT_RECOVERY_RELIABILITY_THRESHOLD,
        structure_effort_by_complexity=settings.GRAPH_STRUCTURE_EFFORT_BY_COMPLEXITY,
        default_reasoning_effort=settings.AGENT_REASONING_EFFORT,
    )


def compile_estimation_graph(
    deps: MultiAgentDeps,
    *,
    checkpointer: Any,
    proposal_enabled: bool = True,
):
    """Factory used by the lifespan to compile the production multi-agent graph."""
    return build_estimation_graph(
        deps, checkpointer=checkpointer, proposal_enabled=proposal_enabled
    )


async def start_graph_run(
    estimation_id: str,
    transcript: str,
    runtime: LangGraphRuntime | None,
) -> GraphRunState:
    """START the multi-agent flow until the first human gate."""
    graph_runtime = _require_runtime(runtime)
    config = {"configurable": {"thread_id": estimation_id}}
    try:
        await graph_runtime.graph.ainvoke(
            {"transcript": transcript, "estimation_id": estimation_id},
            config=config,
        )
        snapshot = await graph_runtime.graph.aget_state(config)
    except Exception as exc:  # noqa: BLE001
        log.error(
            "graph_start_failed",
            estimation_id=estimation_id,
            error_type=type(exc).__name__,
        )
        raise GraphEstimationError("Graph estimation failed.") from exc
    return snapshot_to_run_state(estimation_id, snapshot)


async def resume_graph_run(
    estimation_id: str,
    decision: dict,
    runtime: LangGraphRuntime | None,
) -> GraphRunState:
    """RESUME a paused run with the human's gate decision."""
    graph_runtime = _require_runtime(runtime)
    config = {"configurable": {"thread_id": estimation_id}}
    snapshot = await graph_runtime.graph.aget_state(config)
    if not snapshot.next:
        raise GraphConflictError(
            "No pending human gate for this estimation_id (already completed or unknown)."
        )
    try:
        await graph_runtime.graph.ainvoke(Command(resume=decision), config)
        snapshot = await graph_runtime.graph.aget_state(config)
    except Exception as exc:  # noqa: BLE001
        log.error(
            "graph_resume_failed",
            estimation_id=estimation_id,
            error_type=type(exc).__name__,
        )
        raise GraphEstimationError("Failed to resume the estimate.") from exc
    return snapshot_to_run_state(estimation_id, snapshot)


async def read_graph_state(
    estimation_id: str,
    runtime: LangGraphRuntime | None,
) -> GraphRunState:
    """Read the current snapshot of a run."""
    graph_runtime = _require_runtime(runtime)
    config = {"configurable": {"thread_id": estimation_id}}
    snapshot = await graph_runtime.graph.aget_state(config)
    if not getattr(snapshot, "created_at", None) and not snapshot.values:
        raise GraphNotFoundError("Unknown estimation_id.")
    return snapshot_to_run_state(estimation_id, snapshot)


async def read_graph_progress(
    estimation_id: str,
    runtime: LangGraphRuntime | None,
    activity: GraphActivityLog,
) -> GraphProgress:
    """Poll a background run: state plus the activity feed."""
    graph_runtime = _require_runtime(runtime)
    config = {"configurable": {"thread_id": estimation_id}}
    snapshot = await graph_runtime.graph.aget_state(config)
    run_state = snapshot_to_run_state(estimation_id, snapshot)
    data = run_state.model_dump()
    data["state"] = progress_state(snapshot)
    data["activity"] = [
        ActivityEntry(**entry) for entry in activity.read(estimation_id)
    ]
    return GraphProgress(**data)


async def stream_graph_run(
    *,
    estimation_id: str,
    transcript: str,
    runtime: LangGraphRuntime,
    activity: GraphActivityLog,
    payload: dict | Command | None = None,
) -> None:
    """Background task: drive the graph with ``astream`` and append activity lines."""
    config = {"configurable": {"thread_id": estimation_id}}
    invoke_payload = payload or {
        "transcript": transcript,
        "estimation_id": estimation_id,
    }
    try:
        async for chunk in runtime.graph.astream(
            invoke_payload, config, stream_mode="updates"
        ):
            for node_name, update in chunk.items():
                for line in describe_node(node_name, update):
                    activity.append(
                        estimation_id,
                        node=line["node"],
                        label=line["label"],
                        message=line["message"],
                    )
    except Exception as exc:  # noqa: BLE001
        log.error(
            "graph_stream_failed",
            estimation_id=estimation_id,
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        activity.append(
            estimation_id, node="error", label="Error", message=str(exc)[:200]
        )


async def stream_graph_resume(
    *,
    estimation_id: str,
    decision: dict,
    runtime: LangGraphRuntime,
    activity: GraphActivityLog,
) -> None:
    """Background task: resume a paused run and append activity lines."""
    await stream_graph_run(
        estimation_id=estimation_id,
        transcript="",
        runtime=runtime,
        activity=activity,
        payload=Command(resume=decision),
    )


async def draft_graph_proposal(
    estimation_id: str,
    runtime: LangGraphRuntime | None,
    *,
    propose: Callable[[str], Awaitable[CommercialProposal]],
) -> CommercialProposal:
    """Draft a commercial proposal from the persisted estimate snapshot."""
    graph_runtime = _require_runtime(runtime)
    config = {"configurable": {"thread_id": estimation_id}}
    snapshot = await graph_runtime.graph.aget_state(config)
    estimate = (snapshot.values or {}).get("estimate")
    if not estimate:
        raise GraphConflictError(
            "No validated estimate for this estimation_id (run not far enough / unknown)."
        )
    analysis_report = (snapshot.values or {}).get("analysis_report") or {}
    try:
        return await propose(_proposal_input(estimate, analysis_report))
    except GraphConflictError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.error(
            "graph_proposal_failed",
            estimation_id=estimation_id,
            error_type=type(exc).__name__,
        )
        raise GraphEstimationError("Failed to draft the proposal.") from exc
