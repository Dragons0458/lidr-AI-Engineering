"""Retrieval adapter for task-hours recovery agents."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any

from app.config import get_settings
from app.dependencies import get_embedder
from app.generation.rag.retrieval.collections import Collection
from app.generation.rag.retrieval.pipeline import retrieve

RecoveryBackend = Callable[[str, list[str] | None], Awaitable[list[dict[str, Any]]]]
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_PREVIEW_CHARS = 160


def make_retrieval_backend(
    *,
    top_k: int,
    distance_threshold: float,
    search_mode: str,
    rerank: bool,
) -> RecoveryBackend:
    """Build a resolved retrieval backend for historical task searches."""

    async def backend(query: str, sectors: list[str] | None) -> list[dict[str, Any]]:
        embedder = get_embedder()
        if embedder is None:
            raise RuntimeError("Embedding service is not available.")
        settings = get_settings()
        embedding = await asyncio.to_thread(embedder.embed_one, query)
        result = await retrieve(
            query_embedding=embedding,
            query_text=query,
            collection=Collection.BUDGET,
            chunk_types=["historical_task"],
            sectors=sectors,
            top_k=top_k,
            distance_threshold=distance_threshold,
            search_mode=search_mode,
            rerank=rerank,
            recall_k=settings.RETRIEVAL_RECALL_TOP_K,
            rerank_top_n=top_k,
            rrf_k=settings.RRF_K,
        )
        return [
            {
                "id": chunk.id,
                "content_preview": _CONTROL_CHARS.sub(
                    "", " ".join(chunk.content.split())
                )[:_PREVIEW_CHARS],
                "sector": chunk.sector,
                "budget_id": chunk.budget_id,
                "estimated_hours": int(chunk.estimated_hours),
                "distance": round(chunk.distance, 4),
            }
            for chunk in result.chunks
            if chunk.estimated_hours is not None
        ]

    return backend


def make_default_legacy_recovery_backend() -> RecoveryBackend:
    """Build the Session 12 legacy defaults without importing agentic models."""
    return make_retrieval_backend(
        top_k=5,
        distance_threshold=0.45,
        search_mode="vector",
        rerank=False,
    )
