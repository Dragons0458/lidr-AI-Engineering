"""Semantic retriever over the pgvector store (Session 8).

Embeds the query with the SAME model used at ingest time (mixing embedding
models makes distances meaningless) and ranks chunks by cosine distance via
SQL. Session 10 adds hybrid search and reranking via ``pipeline.retrieve``.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import get_settings
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.schemas import (
    SearchHit,
    SearchResponse,
)
from app.generation.rag.store.repository import ChunkStore

log = structlog.get_logger()


class SemanticRetriever:
    """k-NN retrieval: embed the query, rank chunks by cosine distance."""

    def __init__(
        self,
        embedder: OpenAIEmbedder,
        session_factory: async_sessionmaker,
        store: ChunkStore,
    ) -> None:
        self._embedder = embedder
        self._session_factory = session_factory
        self._store = store

    async def search(self, *, query: str, k: int) -> SearchResponse:
        started = time.perf_counter()

        query_vector = await asyncio.to_thread(self._embedder.embed_one, query)

        async with self._session_factory() as session:
            rows = await self._store.search(session, query_vector=query_vector, k=k)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        response = SearchResponse(
            query=query,
            k=k,
            search_time_ms=elapsed_ms,
            results=[
                SearchHit(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    chunk_type=row.chunk_type,
                    content=row.content,
                    distance=float(row.distance),
                    metadata=row.metadata_,
                )
                for row in rows
            ],
        )
        log.info(
            "rag_search_done",
            query=query[:80],
            k=k,
            results=len(response.results),
            search_time_ms=elapsed_ms,
        )
        return response


async def search_chunks(
    query_embedding: list[float],
    *,
    query_text: str = "",
    top_k: int = 10,
    distance_threshold: float = 0.6,
    sectors: list[str] | None = None,
    project_year_min: int | None = None,
    project_year_max: int | None = None,
    chunk_types: list[str] | None = None,
    search_mode: str | None = None,
    rerank: bool | None = None,
):
    """Metadata-filtered retrieval with optional hybrid search and reranking."""
    from app.generation.rag.retrieval.pipeline import retrieve

    settings = get_settings()
    effective_mode = search_mode or settings.RETRIEVAL_SEARCH_MODE
    effective_rerank = rerank if rerank is not None else settings.RERANKER_ENABLED
    rerank_top_n = settings.RERANK_TOP_N if effective_rerank else top_k

    return await retrieve(
        query_embedding=query_embedding,
        query_text=query_text,
        search_mode=effective_mode,
        rerank=effective_rerank,
        top_k=top_k,
        recall_k=settings.RETRIEVAL_RECALL_TOP_K,
        rerank_top_n=rerank_top_n,
        distance_threshold=distance_threshold,
        rrf_k=settings.RRF_K,
        sectors=sectors,
        project_year_min=project_year_min,
        project_year_max=project_year_max,
        chunk_types=chunk_types,
    )
