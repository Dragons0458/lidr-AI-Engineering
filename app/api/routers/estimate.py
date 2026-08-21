"""``POST /v1/estimate/from-transcript`` — transcript → grounded estimate (S09)."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.rate_limiting import limiter
from app.api.security import require_estimate_key
from app.foundation.observability.usage import set_outcome
from app.generation.rag.errors import RagError
from app.generation.rag.estimator import estimate_from_transcript
from app.generation.rag.schemas import Estimate, EstimateRequest

log = structlog.get_logger()

router = APIRouter(prefix="/v1/estimate", tags=["estimate"])


@router.post(
    "/from-transcript",
    response_model=Estimate,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("10/minute")
async def from_transcript(request: Request, payload: EstimateRequest) -> Estimate:
    """Produce a grounded estimate from a raw transcript (idempotent on key)."""
    try:
        result = await estimate_from_transcript(
            payload.transcript,
            idempotency_key=payload.idempotency_key,
        )
        set_outcome(
            confidence=result.confidence,
            abstained=result.confidence == "insufficient"
            or result.total_engineer_days is None,
        )
        return result
    except RagError as exc:
        log.error(
            "estimate_failed", error_type=type(exc).__name__, error=str(exc)[:300]
        )
        raise HTTPException(
            status_code=502, detail="Failed to produce an estimate."
        ) from exc
