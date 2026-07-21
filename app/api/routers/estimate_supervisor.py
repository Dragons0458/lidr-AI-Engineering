"""``/v1/estimate/agent/supervisor`` — Session 14 multi-agent supervisor flow."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import get_request_id
from app.api.rate_limiting import limiter
from app.api.security import require_estimate_key
from app.domain.schemas.supervisor_estimation import (
    SupervisorEstimateRequest,
    SupervisorResumeRequest,
    SupervisorRunState,
)
from app.domain.supervisor_estimation import (
    SupervisorConflictError,
    SupervisorEstimationError,
    SupervisorNotFoundError,
    SupervisorRuntimeUnavailableError,
    read_supervisor_state,
    resume_supervisor_run,
    start_supervisor_run,
)
from app.generation.rag.observability import log_stage

log = structlog.get_logger()
router = APIRouter(prefix="/v1/estimate/agent", tags=["estimate-agent-supervisor"])


def get_graph_runtime(request: Request):
    """Resolve the lifespan-owned LangGraph runtime (may be ``None``)."""
    return getattr(request.app.state, "graph_runtime", None)


@router.post(
    "/supervisor",
    response_model=SupervisorRunState,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("10/minute")
async def estimate_supervisor(
    request: Request,
    payload: SupervisorEstimateRequest,
) -> SupervisorRunState:
    """START a supervisor run; runs to completion or to the human gate."""
    runtime = get_graph_runtime(request)
    try:
        with log_stage(
            "agent_supervisor_start",
            get_request_id(request),
            estimation_id=payload.estimation_id,
        ):
            return await start_supervisor_run(
                payload.transcript,
                runtime,
                estimation_id=payload.estimation_id,
            )
    except SupervisorRuntimeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SupervisorEstimationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.error(
            "stage_failed",
            stage="agent_supervisor_start",
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=502, detail="Supervisor estimation failed."
        ) from exc


@router.post(
    "/supervisor/{estimation_id}/resume",
    response_model=SupervisorRunState,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("10/minute")
async def resume_supervisor(
    request: Request,
    estimation_id: str,
    payload: SupervisorResumeRequest,
) -> SupervisorRunState:
    """RESUME a paused run with the reviewer's decision."""
    runtime = get_graph_runtime(request)
    try:
        with log_stage(
            "agent_supervisor_resume",
            get_request_id(request),
            estimation_id=estimation_id,
        ):
            return await resume_supervisor_run(
                estimation_id,
                payload.model_dump(),
                runtime,
            )
    except SupervisorRuntimeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SupervisorConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SupervisorEstimationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.error(
            "stage_failed",
            stage="agent_supervisor_resume",
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=502, detail="Failed to resume the estimate."
        ) from exc


@router.get(
    "/supervisor/{estimation_id}/state",
    response_model=SupervisorRunState,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("60/minute")
async def supervisor_state(
    request: Request,
    estimation_id: str,
) -> SupervisorRunState:
    """Read the current snapshot of a supervisor run."""
    runtime = get_graph_runtime(request)
    try:
        return await read_supervisor_state(estimation_id, runtime)
    except SupervisorRuntimeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SupervisorNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.error(
            "stage_failed",
            stage="agent_supervisor_state",
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=502, detail="Failed to read supervisor state."
        ) from exc
