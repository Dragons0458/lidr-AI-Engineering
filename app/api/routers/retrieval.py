"""``POST /v1/retrieval/search`` — metadata-filtered semantic retrieval (S09)."""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.rate_limiting import limiter
from app.api.security import require_retrieval_key
from app.dependencies import get_embedder
from app.generation.rag.errors import RetrievalError
from app.generation.rag.retriever import search_chunks
from app.generation.rag.schemas import RetrievalRequest, RetrievalResult

log = structlog.get_logger()

router = APIRouter(prefix="/v1/retrieval", tags=["retrieval"])


@router.post(
    "/search",
    response_model=RetrievalResult,
    dependencies=[Depends(require_retrieval_key)],
)
@limiter.limit("120/minute")
async def search(request: Request, payload: RetrievalRequest) -> RetrievalResult:
    """Return chunks within ``distance_threshold`` of the embedded query text."""
    embedder = get_embedder()
    if embedder is None:
        log.error("retrieval_failed", reason="embedder_unavailable")
        raise HTTPException(
            status_code=500, detail="Embedding service is not available."
        )

    try:
        query_embedding = await asyncio.to_thread(
            embedder.embed_one, payload.query_text
        )
        return await search_chunks(
            query_embedding,
            query_text=payload.query_text,
            top_k=payload.top_k,
            distance_threshold=payload.distance_threshold,
            sectors=payload.sectors,
            project_year_min=payload.project_year_min,
            project_year_max=payload.project_year_max,
            chunk_types=payload.chunk_types,
        )
    except RetrievalError as exc:
        raise HTTPException(status_code=502, detail="Retrieval failed.") from exc
    except Exception as exc:  # noqa: BLE001
        log.error(
            "retrieval_failed",
            reason="search_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=502, detail="Failed to run retrieval.") from exc
