"""HTTP transport for the Session 13 LangGraph multi-agent estimation flow."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from app.api.deps import get_request_id
from app.api.rate_limiting import limiter
from app.api.security import require_estimate_key
from app.dependencies import get_graph_activity
from app.domain.graph_estimation import (
    GraphConflictError,
    GraphEstimationError,
    GraphNotFoundError,
    GraphRuntimeUnavailableError,
    draft_graph_proposal,
    read_graph_progress,
    read_graph_state,
    resume_graph_run,
    start_graph_run,
    stream_graph_resume,
    stream_graph_run,
)
from app.domain.schemas.graph_estimation import (
    GraphEstimationRequest,
    GraphProgress,
    GraphProposalResponse,
    GraphResumeRequest,
    GraphRunState,
)
from app.generation.agentic.graph.activity import GraphActivityLog
from app.generation.rag.observability import log_stage

log = structlog.get_logger()
router = APIRouter(prefix="/v1/estimate/agent", tags=["estimate-agent-graph"])


def get_graph_runtime(request: Request):
    """Resolve the lifespan-owned LangGraph runtime (may be ``None``)."""
    return getattr(request.app.state, "graph_runtime", None)


def get_multiagent_deps(request: Request):
    """Resolve the lifespan-owned multi-agent dependency bundle."""
    return getattr(request.app.state, "multiagent_deps", None)


@router.post(
    "/graph",
    response_model=GraphRunState,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("10/minute")
async def estimate_graph(
    request: Request,
    payload: GraphEstimationRequest,
) -> GraphRunState:
    """START the multi-agent flow; runs to the first human gate."""
    runtime = get_graph_runtime(request)
    try:
        with log_stage(
            "agent_graph_start",
            get_request_id(request),
            estimation_id=payload.estimation_id,
        ):
            return await start_graph_run(
                payload.estimation_id,
                payload.transcript,
                runtime,
            )
    except GraphRuntimeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GraphEstimationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.error(
            "stage_failed",
            stage="agent_graph_start",
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=502, detail="Graph estimation failed.") from exc


@router.post(
    "/graph/{estimation_id}/resume",
    response_model=GraphRunState,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("10/minute")
async def resume_graph(
    request: Request,
    estimation_id: str,
    payload: GraphResumeRequest,
) -> GraphRunState:
    """RESUME a paused run with the human's decision."""
    runtime = get_graph_runtime(request)
    try:
        with log_stage(
            "agent_graph_resume",
            get_request_id(request),
            estimation_id=estimation_id,
        ):
            return await resume_graph_run(
                estimation_id,
                payload.decision,
                runtime,
            )
    except GraphRuntimeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GraphConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GraphEstimationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.error(
            "stage_failed",
            stage="agent_graph_resume",
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=502, detail="Failed to resume the estimate."
        ) from exc


@router.get(
    "/graph/{estimation_id}/state",
    response_model=GraphRunState,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("60/minute")
async def graph_state(
    request: Request,
    estimation_id: str,
) -> GraphRunState:
    """Read the current snapshot of a run (pending gate + artifacts)."""
    runtime = get_graph_runtime(request)
    try:
        with log_stage(
            "agent_graph_state",
            get_request_id(request),
            estimation_id=estimation_id,
        ):
            return await read_graph_state(estimation_id, runtime)
    except GraphRuntimeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GraphNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.error(
            "stage_failed",
            stage="agent_graph_state",
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=502, detail="Failed to read graph state."
        ) from exc


@router.post(
    "/graph/stream",
    response_model=GraphProgress,
    status_code=202,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("10/minute")
async def estimate_graph_stream(
    request: Request,
    payload: GraphEstimationRequest,
    background: BackgroundTasks,
    activity: GraphActivityLog = Depends(get_graph_activity),
) -> GraphProgress:
    """START the flow in the background; poll ``/progress`` for activity."""
    runtime = get_graph_runtime(request)
    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail="LangGraph estimation runtime is not available.",
        )
    activity.reset(payload.estimation_id)
    background.add_task(
        stream_graph_run,
        estimation_id=payload.estimation_id,
        transcript=payload.transcript,
        runtime=runtime,
        activity=activity,
    )
    log.info(
        "graph_stream_started",
        request_id=get_request_id(request),
        estimation_id=payload.estimation_id,
    )
    return GraphProgress(
        estimation_id=payload.estimation_id, state="running", activity=[]
    )


@router.post(
    "/graph/{estimation_id}/resume-stream",
    response_model=GraphProgress,
    status_code=202,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("10/minute")
async def resume_graph_stream(
    request: Request,
    estimation_id: str,
    payload: GraphResumeRequest,
    background: BackgroundTasks,
    activity: GraphActivityLog = Depends(get_graph_activity),
) -> GraphProgress:
    """RESUME a paused run in the background; poll ``/progress`` for activity."""
    runtime = get_graph_runtime(request)
    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail="LangGraph estimation runtime is not available.",
        )
    try:
        snapshot = await runtime.graph.aget_state(
            {"configurable": {"thread_id": estimation_id}}
        )
        if not snapshot.next:
            raise GraphConflictError(
                "No pending human gate for this estimation_id (already completed or unknown)."
            )
    except GraphConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    background.add_task(
        stream_graph_resume,
        estimation_id=estimation_id,
        decision=payload.decision,
        runtime=runtime,
        activity=activity,
    )
    log.info(
        "graph_resume_stream_started",
        request_id=get_request_id(request),
        estimation_id=estimation_id,
    )
    return GraphProgress(estimation_id=estimation_id, state="running", activity=[])


@router.get(
    "/graph/{estimation_id}/progress",
    response_model=GraphProgress,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("120/minute")
async def graph_progress(
    request: Request,
    estimation_id: str,
    activity: GraphActivityLog = Depends(get_graph_activity),
) -> GraphProgress:
    """Poll a background run: current state plus the activity feed."""
    runtime = get_graph_runtime(request)
    try:
        return await read_graph_progress(estimation_id, runtime, activity)
    except GraphRuntimeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.error(
            "stage_failed",
            stage="agent_graph_progress",
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=502, detail="Failed to read graph progress."
        ) from exc


@router.post(
    "/graph/{estimation_id}/proposal",
    response_model=GraphProposalResponse,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("10/minute")
async def graph_proposal(
    request: Request,
    estimation_id: str,
) -> GraphProposalResponse:
    """Draft or re-draft the commercial proposal from the validated estimate."""
    runtime = get_graph_runtime(request)
    deps = get_multiagent_deps(request)
    if deps is None:
        raise HTTPException(
            status_code=503,
            detail="LangGraph estimation runtime is not available.",
        )
    try:
        with log_stage(
            "agent_graph_proposal",
            get_request_id(request),
            estimation_id=estimation_id,
        ):
            proposal = await draft_graph_proposal(
                estimation_id,
                runtime,
                propose=deps.propose,
            )
    except GraphRuntimeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GraphConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GraphEstimationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.error(
            "stage_failed",
            stage="agent_graph_proposal",
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=502, detail="Failed to draft the proposal."
        ) from exc

    return GraphProposalResponse(
        estimation_id=estimation_id,
        title=proposal.title,
        executive_summary=proposal.executive_summary,
        scope=proposal.scope,
        total_engineer_days=proposal.total_engineer_days,
        body_markdown=proposal.body_markdown,
    )
