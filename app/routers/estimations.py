import asyncio
from collections.abc import AsyncIterator
from typing import Literal

import structlog
from fastapi import APIRouter, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.errors.llm_error import LLMServiceError
from app.formatters.llm_formatters import format_response
from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.services.estimation_service import (
    generate_estimation,
    generate_estimation_stream,
)
from app.services.evaluation import evaluate_estimation_structure

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
        response = format_response(
            generate_estimation(request, prompt_version=prompt_version),
            prompt_version=prompt_version,
        )
        if request.evaluate:
            response.validation = evaluate_estimation_structure(
                response.estimation,
                response.finish_reason,
                request.output_format,
            )
        return response
    except LLMServiceError as e:
        log.error("estimation_endpoint_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/estimate/stream")
async def estimate_stream(
    request: EstimationRequest,
    prompt_version: Literal["v1", "v2"] = Query(default="v1"),
) -> EventSourceResponse:
    """Stream estimation text via Server-Sent Events (token / done / error)."""

    model = request.model or settings.PRIMARY_MODEL

    async def event_generator() -> AsyncIterator[dict]:
        loop = asyncio.get_running_loop()
        chunks = generate_estimation_stream(
            request,
            prompt_version=prompt_version,
            use_cache=True,
        )

        def _next_chunk() -> str | None:
            try:
                return next(chunks)
            except StopIteration:
                return None
            except Exception as exc:
                log.error(
                    "estimate_stream_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                raise

        yield {
            "event": "meta",
            "data": f'{{"model":"{model}","prompt_version":"{prompt_version}"}}',
        }

        try:
            while True:
                chunk = await loop.run_in_executor(None, _next_chunk)
                if chunk is None:
                    break
                if chunk:
                    yield {"event": "token", "data": chunk}
            yield {"event": "done", "data": "[DONE]"}
        except LLMServiceError as exc:
            yield {"event": "error", "data": str(exc)}
        except Exception as exc:
            yield {"event": "error", "data": str(exc)}

    try:
        return EventSourceResponse(event_generator())
    except LLMServiceError as e:
        log.error("estimation_stream_endpoint_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
