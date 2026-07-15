"""HTTP transport for the Session 12 hybrid estimation agent."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import get_request_id
from app.api.rate_limiting import limiter
from app.api.security import require_estimate_key
from app.config import get_settings
from app.dependencies import (
    get_async_openai_client,
    get_embedder,
    get_runtime_retrieval_config,
)
from app.domain.agent_estimation import (
    OpenAIClientMissingError,
    RecoveryAgentError,
    agent_estimate_task_hours,
    agent_propose_structure,
)
from app.generation.rag.observability import log_stage
from app.generation.rag.schemas import (
    AgentHoursRequest,
    AgentStructureRequest,
    GenerateResult,
    TaskHoursResult,
)

log = structlog.get_logger()
router = APIRouter(prefix="/v1/estimate/agent", tags=["estimate-agent"])


@router.post(
    "/structure",
    response_model=GenerateResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("15/minute")
async def structure(request: Request, payload: AgentStructureRequest) -> GenerateResult:
    """Propose a module/task structure without tools, sources, or hours."""
    client = get_async_openai_client()
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client is not available.")
    settings = get_settings()
    try:
        with log_stage("agent_structure", get_request_id(request)):
            return await agent_propose_structure(
                payload.query,
                client=client,
                model=payload.model or settings.AGENT_MODEL,
                reasoning_effort=(
                    payload.reasoning_effort or settings.AGENT_REASONING_EFFORT
                ),
                persona=payload.persona,
            )
    except Exception as exc:  # noqa: BLE001
        log.error(
            "stage_failed", stage="agent_structure", error_type=type(exc).__name__
        )
        raise HTTPException(status_code=502, detail="Agent structure failed.") from exc


@router.post(
    "/hours",
    response_model=TaskHoursResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("15/minute")
async def hours(request: Request, payload: AgentHoursRequest) -> TaskHoursResult:
    """Estimate every task deterministically and recover only flagged rows."""
    if get_embedder() is None:
        raise HTTPException(
            status_code=500, detail="Embedding service is not available."
        )
    settings = get_settings()
    runtime = get_runtime_retrieval_config()
    top_k = payload.search_top_k or runtime.effective_task_hours_top_k()
    threshold = (
        payload.search_distance_threshold
        if payload.search_distance_threshold is not None
        else runtime.effective_task_hours_distance_threshold()
    )
    task_count = sum(len(module.tasks) for module in payload.modules)
    try:
        with log_stage(
            "agent_hours",
            get_request_id(request),
            tasks=task_count,
        ):
            return await agent_estimate_task_hours(
                payload.modules,
                client=get_async_openai_client(),
                model=payload.model or settings.AGENT_MODEL,
                reasoning_effort=(
                    payload.reasoning_effort or settings.AGENT_REASONING_EFFORT
                ),
                max_iterations=(
                    payload.max_iterations or settings.AGENT_MAX_ITERATIONS
                ),
                top_k=top_k,
                distance_threshold=threshold,
                search_mode=runtime.effective_search_mode(),
                rerank=runtime.effective_rerank(),
                persona=payload.persona,
                recovery_reliability_threshold=(
                    settings.AGENT_RECOVERY_RELIABILITY_THRESHOLD
                ),
            )
    except OpenAIClientMissingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RecoveryAgentError as exc:
        raise HTTPException(status_code=502, detail="Agent recovery failed.") from exc
    except Exception as exc:  # noqa: BLE001
        log.error("stage_failed", stage="agent_hours", error_type=type(exc).__name__)
        raise HTTPException(
            status_code=502, detail="Task-hours estimation failed."
        ) from exc
