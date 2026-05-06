import structlog
from fastapi import APIRouter, HTTPException

from app.errors import llm_error
from app.formatters.llm_formatters import format_response
from app.schemas.estimation import EstimationRequest, EstimationResponse
from app.services.estimation_service import generate_estimation

log = structlog.get_logger()

router = APIRouter(tags=["estimations"])


@router.post("/estimate", response_model=EstimationResponse)
async def estimate(request: EstimationRequest) -> EstimationResponse:
    """Generates an estimation for a software development project based on a meeting summary."""

    try:
        return format_response(generate_estimation(request.transcript))
    except llm_error as e:
        log.error("estimation_endpoint_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
