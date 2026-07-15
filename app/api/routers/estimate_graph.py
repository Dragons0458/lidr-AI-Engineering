"""HTTP transport for the Session 13 LangGraph estimation endpoint."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import get_request_id
from app.api.rate_limiting import limiter
from app.api.security import require_estimate_key
from app.domain.graph_estimation import (
    GraphEstimationError,
    GraphRuntimeUnavailableError,
    run_graph_estimation,
)
from app.domain.schemas.graph_estimation import (
    GraphEstimationRequest,
    GraphEstimationResponse,
)
from app.generation.rag.observability import log_stage

log = structlog.get_logger()
router = APIRouter(prefix="/v1/estimate/agent", tags=["estimate-agent-graph"])


def get_graph_runtime(request: Request):
    """Resolve the lifespan-owned LangGraph runtime (may be ``None``)."""
    return getattr(request.app.state, "graph_runtime", None)


@router.post(
    "/graph",
    response_model=GraphEstimationResponse,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("15/minute")
async def estimate_graph(
    request: Request,
    payload: GraphEstimationRequest,
) -> GraphEstimationResponse:
    """Run the sequential LangGraph estimation workflow end-to-end."""
    runtime = get_graph_runtime(request)
    try:
        with log_stage(
            "agent_graph",
            get_request_id(request),
            estimation_id=payload.estimation_id,
        ):
            return await run_graph_estimation(
                estimation_id=payload.estimation_id,
                transcript=payload.transcript,
                runtime=runtime,
            )
    except GraphRuntimeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GraphEstimationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.error(
            "stage_failed",
            stage="agent_graph",
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=502, detail="Graph estimation failed.") from exc
