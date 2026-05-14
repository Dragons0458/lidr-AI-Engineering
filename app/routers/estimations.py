from typing import Literal

import structlog
from fastapi import APIRouter, HTTPException, Query
from starlette.responses import StreamingResponse

from app.errors.llm_error import LLMServiceError
from app.formatters.llm_formatters import format_response
from app.config import get_settings
from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.services.estimation_service import generate_estimation, generate_estimation_stream

log = structlog.get_logger()
settings = get_settings()

router = APIRouter(tags=["estimations"])


@router.post("/estimate", response_model=EstimationResponse)
async def estimate(
    request: EstimationRequest,
    prompt_version: Literal["v1", "v2"] = Query(default="v1"),
) -> EstimationResponse:
    """Generates an estimation for a software development project based on a meeting summary."""

    try:
        return format_response(
            generate_estimation(request, prompt_version=prompt_version),
            prompt_version=prompt_version,
        )
    except LLMServiceError as e:
        log.error("estimation_endpoint_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/estimate/stream")
async def estimate_stream(
    request: EstimationRequest,
    prompt_version: Literal["v1", "v2"] = Query(default="v1"),
) -> StreamingResponse:
    """Streams estimation text for a software development project."""

    try:
        return StreamingResponse(
            generate_estimation_stream(request, prompt_version=prompt_version),
            media_type="text/plain; charset=utf-8",
            headers={
                "X-LLM-Model": settings.LLM_MODEL,
                "X-Prompt-Version": prompt_version,
            },
        )
    except LLMServiceError as e:
        log.error("estimation_stream_endpoint_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
