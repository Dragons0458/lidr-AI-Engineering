"""Domain conductor for the Session 14 supervisor multi-agent estimation flow."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import structlog
from langgraph.types import Command

from app.domain.schemas.supervisor_estimation import (
    PendingHumanReview,
    SupervisorRunState,
)
from app.foundation.persistence.langgraph import LangGraphRuntime
from app.generation.agentic.agent_loop import run_structure_agent
from app.generation.agentic.agent_schemas import AgentStructure
from app.generation.agentic.agent_tools import adapt_recovery_backend
from app.generation.agentic.graph.supervisor_build import build_supervisor_graph
from app.generation.agentic.graph.supervisor_nodes import (
    SupervisorDecision,
    SupervisorDeps,
)
from app.generation.agentic.graph.supervisor_state import privilege_violations
from app.generation.rag.agent_retrieval import make_retrieval_backend
from app.generation.rag.prompt_builder import build_structure_user_message
from app.generation.rag.query_reformulator import reformulate_query
from app.generation.rag.schemas import EstimationQuery

log = structlog.get_logger()

_ROUTER_SYSTEM_PROMPT = """\
You are the SUPERVISOR of a software-estimation multi-agent system. You do not do any \
estimation work yourself: you read the current state digest and decide which specialist \
acts next.

The specialists and what each one produces:

- requirements_extractor: reads the transcript and produces requirements + components. \
No tools. Needs: the transcript.
- budget_searcher: searches historical budgets and produces budget_matches. \
Needs: components. Tool: search_budgets.
- estimate_generator: turns references into a consolidated hours estimate. \
Needs: components and a completed search (matches may be empty). Tool: calculate_estimate.
- coherence_validator: runs guardrails and publishes confidence / out_of_range facts. \
Needs: estimate. Tool: validate_estimate.

Rules:
- Choose the ONE agent that can make progress right now.
- Never choose an agent whose inputs do not exist yet.
- Never re-run an agent that already acted.
- Choose "finish" once the estimate has been produced AND validated.
- Explain your choice in one line.
"""

_CANONICAL_RETRIEVAL_MODULES = (
    "Admin & Back-office",
    "Analytics & Reporting",
    "Authentication & Access",
    "Data & Integrations",
    "Frontend / UX",
    "Infrastructure & DevOps",
    "Integrations Platform",
    "Operations",
    "Records & Compliance",
    "Security & Compliance",
    "Telemetry & IoT",
)


class SupervisorRuntimeUnavailableError(RuntimeError):
    """Raised when the supervisor graph was not compiled at startup."""


class SupervisorEstimationError(RuntimeError):
    """Raised when the supervisor run fails for a technical reason."""


class SupervisorConflictError(RuntimeError):
    """Raised when resume is requested but nothing is pending."""


class SupervisorNotFoundError(RuntimeError):
    """Raised when no checkpoint exists for the given estimation_id."""


def _thread_id(estimation_id: str) -> str:
    return f"s14:{estimation_id}"


def thread_config(estimation_id: str) -> dict:
    """Public helper: namespaced LangGraph thread config for S14 runs."""
    return {"configurable": {"thread_id": _thread_id(estimation_id)}}


def _require_supervisor_graph(runtime: LangGraphRuntime | None) -> Any:
    if runtime is None or getattr(runtime, "supervisor_graph", None) is None:
        raise SupervisorRuntimeUnavailableError("Supervisor graph is not available.")
    return runtime.supervisor_graph


def snapshot_to_run_state(estimation_id: str, snapshot: Any) -> SupervisorRunState:
    """Turn a LangGraph ``StateSnapshot`` into the public ``SupervisorRunState``."""
    values = snapshot.values or {}
    paused = bool(snapshot.next)
    interrupts = getattr(snapshot, "interrupts", None) or ()

    pending_review = None
    if paused and interrupts:
        payload = interrupts[0].value or {}
        pending_review = PendingHumanReview(
            gate=payload.get("gate", "low_confidence_review"),
            estimation_id=estimation_id,
            reasons=payload.get("reasons") or [],
            confidence=payload.get("confidence"),
            threshold=payload.get("threshold"),
            estimate=payload.get("estimate"),
            validation=payload.get("validation"),
            risk_flags=payload.get("risk_flags") or [],
        )

    status = (
        "awaiting_human_review"
        if pending_review is not None
        else (values.get("status") or "needs_review")
    )

    return SupervisorRunState(
        estimation_id=estimation_id,
        state="paused" if paused else "completed",
        status=status,
        pending_review=pending_review,
        estimate=values.get("estimate"),
        confidence=values.get("confidence"),
        requirements=values.get("requirements") or [],
        components=values.get("components") or [],
        budget_matches=values.get("budget_matches") or [],
        validation=values.get("validation"),
        risk_flags=values.get("risk_flags") or [],
        human_decision=values.get("human_decision"),
        routing_history=values.get("routing_history") or [],
        agent_contributions=values.get("agent_contributions") or [],
        privilege_violations=privilege_violations(values),
        errors=values.get("errors") or [],
    )


def build_default_supervisor_deps(
    *,
    client: Any,
    settings: Any,
    llm_wrapper: Any | None = None,
    runtime_retrieval: Any | None = None,
) -> SupervisorDeps:
    """Wire production adapters for the supervisor graph."""
    from app.dependencies import get_llm_wrapper

    wrapper = llm_wrapper or get_llm_wrapper()
    search_mode = "hybrid"
    rerank = True
    if runtime_retrieval is not None:
        search_mode = runtime_retrieval.effective_search_mode()
        rerank = runtime_retrieval.effective_rerank()

    async def reformulate(transcript: str) -> EstimationQuery:
        return await reformulate_query(transcript)

    async def propose_structure(brief: EstimationQuery) -> AgentStructure:
        user_message = build_structure_user_message(brief)
        user_message += (
            "\n\nKeep at most 5 tasks total across at most 4 modules.\n"
            "If the brief is ordinary enterprise software (supplier portal, "
            "CRUD, REST API, web UI, login/roles, ERP sync, basic reports), "
            "name tasks with those standard work-package labels so historical "
            "analogs can match.\n"
            "If the brief describes exotic or novel tech (QKD, COBOL "
            "mainframes, biometric HSM, undocumented protocols, satellite "
            "telemetry at extreme scale, etc.), KEEP those exact domain terms "
            "in task names — do NOT rewrite them into generic CRUD/API labels.\n"
            "For ordinary work, choose the closest exact module label from this "
            "historical-corpus taxonomy: "
            + ", ".join(_CANONICAL_RETRIEVAL_MODULES)
            + "."
        )
        structure, _trace = await run_structure_agent(
            user_message,
            client=client,
            model=settings.AGENT_MODEL,
            reasoning_effort=settings.AGENT_REASONING_EFFORT,
        )
        return structure

    # Recall can be a bit looser than S12 so ordinary packages still retrieve
    # neighbors (~0.5). HITL grounding stays at the configured threshold so
    # distant false neighbours (typical on exotic briefs) do not fake precedent.
    grounding_distance = float(settings.SUPERVISOR_GROUNDING_MAX_DISTANCE)
    search_distance = max(
        float(settings.AGENT_SEARCH_DISTANCE_THRESHOLD),
        grounding_distance,
        0.58,
    )
    recovery = make_retrieval_backend(
        top_k=max(int(settings.AGENT_SEARCH_TOP_K), 8),
        distance_threshold=search_distance,
        search_mode=search_mode,
        rerank=rerank,
    )

    async def route_with_model(digest: str) -> SupervisorDecision:
        decision, _meta = await asyncio.to_thread(
            wrapper.complete_structured,
            system_prompt=_ROUTER_SYSTEM_PROMPT,
            user_message=digest,
            response_model=SupervisorDecision,
            model_override=settings.SUPERVISOR_ROUTER_MODEL,
        )
        return decision

    return SupervisorDeps(
        reformulate=reformulate,
        propose_structure=propose_structure,
        retrieval_backend=adapt_recovery_backend(recovery),
        route_with_model=route_with_model,
        confidence_threshold=settings.SUPERVISOR_CONFIDENCE_THRESHOLD,
        min_grounded_ratio=settings.SUPERVISOR_MIN_GROUNDED_RATIO,
        out_of_range_factor=settings.SUPERVISOR_OUT_OF_RANGE_FACTOR,
        max_steps=settings.SUPERVISOR_MAX_STEPS,
        privilege_strict=settings.SUPERVISOR_PRIVILEGE_STRICT,
        audit_preview_chars=settings.SUPERVISOR_AUDIT_ARGS_PREVIEW_CHARS,
        grounding_max_distance=grounding_distance,
    )


def compile_supervisor_graph(
    deps: SupervisorDeps,
    *,
    checkpointer: Any,
):
    """Factory used by the lifespan to compile the supervisor graph."""
    return build_supervisor_graph(deps, checkpointer=checkpointer)


async def start_supervisor_run(
    transcript: str,
    runtime: LangGraphRuntime | None,
    *,
    estimation_id: str | None = None,
) -> SupervisorRunState:
    """START a supervisor run until completion or the human gate."""
    graph = _require_supervisor_graph(runtime)
    estimation_id = estimation_id or str(uuid4())
    config = {"configurable": {"thread_id": _thread_id(estimation_id)}}
    try:
        await graph.ainvoke(
            {"transcript": transcript, "estimation_id": estimation_id},
            config=config,
        )
        snapshot = await graph.aget_state(config)
    except Exception as exc:  # noqa: BLE001
        log.error(
            "supervisor_start_failed",
            estimation_id=estimation_id,
            error_type=type(exc).__name__,
        )
        raise SupervisorEstimationError("Failed to produce an estimate.") from exc
    return snapshot_to_run_state(estimation_id, snapshot)


async def resume_supervisor_run(
    estimation_id: str,
    decision: dict,
    runtime: LangGraphRuntime | None,
) -> SupervisorRunState:
    """RESUME a paused supervisor run with the reviewer's decision."""
    graph = _require_supervisor_graph(runtime)
    config = {"configurable": {"thread_id": _thread_id(estimation_id)}}
    snapshot = await graph.aget_state(config)
    if not snapshot.next:
        raise SupervisorConflictError(
            "No pending human review for this estimation_id "
            "(already completed or unknown)."
        )
    try:
        await graph.ainvoke(Command(resume=decision), config)
        snapshot = await graph.aget_state(config)
    except Exception as exc:  # noqa: BLE001
        log.error(
            "supervisor_resume_failed",
            estimation_id=estimation_id,
            error_type=type(exc).__name__,
        )
        raise SupervisorEstimationError("Failed to resume the estimate.") from exc
    return snapshot_to_run_state(estimation_id, snapshot)


async def read_supervisor_state(
    estimation_id: str,
    runtime: LangGraphRuntime | None,
) -> SupervisorRunState:
    """Read the current snapshot of a supervisor run."""
    graph = _require_supervisor_graph(runtime)
    config = {"configurable": {"thread_id": _thread_id(estimation_id)}}
    snapshot = await graph.aget_state(config)
    if not getattr(snapshot, "created_at", None) and not snapshot.values:
        raise SupervisorNotFoundError("Unknown estimation_id.")
    return snapshot_to_run_state(estimation_id, snapshot)
