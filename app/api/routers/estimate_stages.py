"""``POST /v1/estimate/stages/*`` — the RAG pipeline, one stage at a time (S09/S10)."""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import get_request_id
from app.api.rate_limiting import limiter
from app.api.security import require_estimate_key
from app.config import get_settings
from app.dependencies import get_embedder, get_token_encoder
from app.generation.rag.context_assembler import (
    build_context_block,
    truncate_to_token_budget,
)
from app.generation.rag.errors import RagError, RetrievalError
from app.generation.rag.estimator import generate_estimate, generate_structure
from app.generation.rag.observability import log_stage
from app.generation.rag.query_reformulator import compose_search_text, reformulate_query
from app.generation.rag.retriever import search_chunks
from app.generation.rag.schemas import (
    AssembleRequest,
    AssembleResult,
    GenerateRequest,
    GenerateResult,
    ReformulateRequest,
    ReformulationResult,
    RetrievalRequest,
    RetrievalResult,
    StructureRequest,
)
from app.generation.rag.validation import check_coherence, validate_citations

log = structlog.get_logger()

router = APIRouter(prefix="/v1/estimate/stages", tags=["estimate-stages"])


@router.post(
    "/reformulate",
    response_model=ReformulationResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("30/minute")
async def reformulate(
    request: Request, payload: ReformulateRequest
) -> ReformulationResult:
    """Stage 1 — distill a transcript into a structured brief + search text."""
    request_id = get_request_id(request)
    try:
        with log_stage("reformulation", request_id):
            query = await reformulate_query(payload.transcript)
            search_text = compose_search_text(query)
        return ReformulationResult(query=query, search_text=search_text)
    except RagError as exc:
        log.error("stage_failed", stage="reformulation", error_type=type(exc).__name__)
        raise HTTPException(
            status_code=502, detail="Query reformulation failed."
        ) from exc


@router.post(
    "/retrieve",
    response_model=RetrievalResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("60/minute")
async def retrieve(request: Request, payload: RetrievalRequest) -> RetrievalResult:
    """Stage 2 — embed the search text and run metadata-filtered k-NN."""
    request_id = get_request_id(request)
    embedder = get_embedder()
    if embedder is None:
        log.error("stage_failed", stage="retrieval", reason="embedder_unavailable")
        raise HTTPException(
            status_code=500, detail="Embedding service is not available."
        )

    try:
        with log_stage("retrieval", request_id, sectors=payload.sectors):
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
                search_mode=payload.search_mode,
                rerank=payload.rerank,
            )
    except RetrievalError as exc:
        raise HTTPException(status_code=502, detail="Retrieval failed.") from exc
    except Exception as exc:  # noqa: BLE001
        log.error("stage_failed", stage="retrieval", error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail="Failed to run retrieval.") from exc


@router.post(
    "/assemble",
    response_model=AssembleResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("60/minute")
async def assemble(request: Request, payload: AssembleRequest) -> AssembleResult:
    """Stage 3 — truncate to the token budget and build the context block."""
    request_id = get_request_id(request)
    settings = get_settings()
    budget = payload.max_context_tokens or settings.MAX_CONTEXT_TOKENS
    encoder = get_token_encoder()

    with log_stage("augmentation", request_id, budget=budget):
        kept = truncate_to_token_budget(payload.chunks, budget, encoder)
        context_block = build_context_block(kept)
        token_count = len(encoder.encode(context_block))

    return AssembleResult(
        context_block=context_block,
        kept_chunks=kept,
        dropped_count=len(payload.chunks) - len(kept),
        token_count=token_count,
    )


@router.post(
    "/structure",
    response_model=GenerateResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("15/minute")
async def structure(request: Request, payload: StructureRequest) -> GenerateResult:
    """Session 10 — generate module→task structure without hours or sources."""
    request_id = get_request_id(request)
    try:
        with log_stage("structure", request_id):
            estimate = await generate_structure(payload.query)
    except RagError as exc:
        log.error("stage_failed", stage="structure", error_type=type(exc).__name__)
        raise HTTPException(
            status_code=502, detail="Structure generation failed."
        ) from exc

    return GenerateResult(estimate=estimate, fabricated_source_ids=[], coherent=True)


@router.post(
    "/generate",
    response_model=GenerateResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("15/minute")
async def generate(request: Request, payload: GenerateRequest) -> GenerateResult:
    """Stage 4 — generate the grounded estimate and report grounding signals."""
    request_id = get_request_id(request)
    try:
        with log_stage(
            "generation",
            request_id,
            sources=len(payload.kept_chunks),
            include_hours=payload.include_hours,
        ):
            estimate = await generate_estimate(
                payload.context_block,
                structured_query=payload.query,
                include_hours=payload.include_hours,
            )
    except RagError as exc:
        log.error("stage_failed", stage="generation", error_type=type(exc).__name__)
        raise HTTPException(
            status_code=502, detail="Estimate generation failed."
        ) from exc

    fabricated = validate_citations(estimate, payload.kept_chunks)
    coherent = check_coherence(estimate)
    return GenerateResult(
        estimate=estimate,
        fabricated_source_ids=fabricated,
        coherent=coherent,
    )
