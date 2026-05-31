from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_embedder
from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder, estimate_embedding_cost_usd
from app.embedding_pipeline.schemas import (
    IngestRequest,
    IngestResponse,
    IngestStats,
)

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1/embeddings", tags=["embeddings"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_embeddings(
    request: IngestRequest,
    embedder: OpenAIEmbedder = Depends(get_embedder),
) -> IngestResponse:
    chunker = JSONStructuralChunker(model=embedder.model)
    chunks = chunker.chunk(request.budgets)
    try:
        embedded_chunks = embedder.embed_many(chunks)
    except Exception as exc:
        log.error(
            "embedding_generation_failed",
            error=str(exc)[:400],
            total_chunks=len(chunks),
        )
        raise HTTPException(
            status_code=500,
            detail="embedding generation failed",
        ) from exc

    total_tokens = sum(chunk.token_count for chunk in chunks)
    return IngestResponse(
        chunks=embedded_chunks,
        stats=IngestStats(
            total_budgets=len(request.budgets),
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            estimated_cost_usd=estimate_embedding_cost_usd(total_tokens),
        ),
    )
