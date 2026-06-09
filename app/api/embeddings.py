"""HTTP layer for the embedding pipeline.

Thin router: it orchestrates chunker -> embedder -> persistence/search and maps
failures to status codes. No business logic lives here.
"""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import ALL_STRATEGIES, build_chunkers, get_chunker, get_embedder
from app.foundation.persistence.async_database import get_async_session
from app.generation.rag.analysis.comparison import (
    ChunkingComparator,
    CompareRequest,
    CompareResponse,
)
from app.generation.rag.chunking.structural import JSONStructuralChunker
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.schemas import (
    Budget,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.generation.rag.store.models import ChunkRow, DocumentRow, EMBEDDING_DIM

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    session: AsyncSession = Depends(get_async_session),
    chunker: JSONStructuralChunker = Depends(get_chunker),
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
) -> IngestResponse:
    """Chunk a budget, embed every chunk, and persist document + vectors in Postgres."""
    if embedder is None:
        log.error("embeddings_ingest_failed", reason="embedder_unavailable")
        raise HTTPException(
            status_code=500, detail="Embedding service is not available."
        )

    started = time.perf_counter()

    existing = await session.scalar(
        select(DocumentRow.id).where(DocumentRow.source_path == request.source_path)
    )
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "detail": "Document already ingested",
                "document_id": existing,
            },
        )

    budget = Budget.model_validate(request.content)
    chunks = chunker.chunk([budget])
    log.info(
        "embeddings_ingest_received",
        source_path=request.source_path,
        total_chunks=len(chunks),
    )

    try:
        embedded = await run_in_threadpool(embedder.embed_many, chunks)
    except Exception as exc:  # noqa: BLE001 — any embedding-API failure becomes a 500.
        log.error(
            "embeddings_ingest_failed",
            reason="embedding_api_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=500, detail="Failed to generate embeddings."
        ) from exc

    doc = DocumentRow(
        source_path=request.source_path,
        document_type=request.document_type,
        doc_metadata={
            "budget_id": budget.budget_id,
            "sector": budget.client_metadata.sector,
            "year": budget.year,
        },
    )
    session.add(doc)
    await session.flush()

    session.add_all(
        [
            ChunkRow(
                document_id=doc.id,
                chunk_type="budget_component",
                content=chunk.text,
                embedding=chunk.embedding,
                chunk_metadata=chunk.metadata,
            )
            for chunk in embedded
        ]
    )
    await session.commit()

    ingestion_time_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "embeddings_ingest_done",
        document_id=doc.id,
        chunks_created=len(embedded),
        ingestion_time_ms=ingestion_time_ms,
    )
    return IngestResponse(
        document_id=doc.id,
        chunks_created=len(embedded),
        embedding_dimension=EMBEDDING_DIM,
        ingestion_time_ms=ingestion_time_ms,
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    session: AsyncSession = Depends(get_async_session),
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
) -> SearchResponse:
    """Semantic search over persisted chunks using cosine distance in SQL."""
    if embedder is None:
        log.error("embeddings_search_failed", reason="embedder_unavailable")
        raise HTTPException(
            status_code=500, detail="Embedding service is not available."
        )

    started = time.perf_counter()

    try:
        query_vector = await run_in_threadpool(embedder.embed_one, request.query)
    except Exception as exc:  # noqa: BLE001
        log.error(
            "embeddings_search_failed",
            reason="embedding_api_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=500, detail="Failed to embed search query."
        ) from exc

    distance_expr = ChunkRow.embedding.cosine_distance(query_vector)
    stmt = (
        select(
            ChunkRow.id,
            ChunkRow.document_id,
            ChunkRow.chunk_type,
            ChunkRow.content,
            ChunkRow.chunk_metadata,
            distance_expr.label("distance"),
        )
        .order_by(distance_expr)
        .limit(request.k)
    )
    rows = (await session.execute(stmt)).all()

    search_time_ms = int((time.perf_counter() - started) * 1000)
    results = [
        SearchResult(
            chunk_id=row.id,
            document_id=row.document_id,
            chunk_type=row.chunk_type,
            content=row.content,
            distance=float(row.distance),
            metadata=row.chunk_metadata,
        )
        for row in rows
    ]
    log.info(
        "embeddings_search_done",
        query=request.query[:80],
        k=request.k,
        hits=len(results),
        search_time_ms=search_time_ms,
    )
    return SearchResponse(
        query=request.query,
        k=request.k,
        search_time_ms=search_time_ms,
        results=results,
    )


@router.post("/compare", response_model=CompareResponse)
def compare(
    request: CompareRequest,
    embedder: OpenAIEmbedder | None = Depends(get_embedder),
) -> CompareResponse:
    """Run several chunking strategies over the same budgets and compare them.

    Returns per-strategy corpus stats and, if queries are given, the top-k
    chunks each strategy retrieves. Nothing is persisted (in-memory only).
    """
    if embedder is None:
        log.error("embeddings_compare_failed", reason="embedder_unavailable")
        raise HTTPException(
            status_code=500, detail="Embedding service is not available."
        )

    names = request.strategies or ALL_STRATEGIES
    try:
        chunkers = build_chunkers(names)
    except KeyError as exc:
        raise HTTPException(
            status_code=400, detail=f"Unknown strategy: {exc.args[0]}"
        ) from exc
    except RuntimeError as exc:
        log.error("embeddings_compare_failed", reason="missing_api_key", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    comparator = ChunkingComparator(chunkers, embedder)
    log.info(
        "embeddings_compare_received",
        total_budgets=len(request.budgets),
        strategies=names,
        n_queries=len(request.queries),
    )
    try:
        stats = comparator.compute_stats(request.budgets)
        queries = comparator.run_queries(
            request.budgets, request.queries, request.top_k
        )
    except Exception as exc:  # noqa: BLE001 — any chunker/embedding failure becomes a 500.
        log.error(
            "embeddings_compare_failed",
            reason="comparison_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=500, detail="Failed to run chunking comparison."
        ) from exc

    return CompareResponse(stats_per_strategy=stats, queries_per_strategy=queries)
