"""Incremental corpus expansion via the existing ingest service (Session 11)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import structlog

from app.generation.rag.ingest_service import DuplicateDocumentError, RagIngestService
from app.generation.rag.schemas import Budget

log = structlog.get_logger()

ProgressCallback = Callable[[int], Awaitable[None] | None]


@dataclass(frozen=True)
class CorpusExpansionResult:
    documents_indexed: int
    documents_skipped: int
    chunks_created: int


class CorpusIndexService:
    """Expand the vector corpus without reimplementing ingestion."""

    def __init__(self, ingest: RagIngestService) -> None:
        self._ingest = ingest

    async def expand(
        self,
        documents: list[Budget],
        *,
        document_type: str,
        chunk_type: str,
        source_prefix: str = "corpus-expansion",
        on_progress: ProgressCallback | None = None,
    ) -> CorpusExpansionResult:
        indexed = skipped = chunks_created = 0

        for budget in documents:
            source_path = f"{source_prefix}::{budget.budget_id}"
            try:
                response = await self._ingest.ingest(
                    source_path=source_path,
                    document_type=document_type,
                    budget=budget,
                    chunk_type=chunk_type,
                )
                indexed += 1
                chunks_created += response.chunks_created
            except DuplicateDocumentError:
                skipped += 1
                log.info("corpus_expansion_skip_duplicate", budget_id=budget.budget_id)
            if on_progress is not None:
                result = on_progress(indexed + skipped)
                if result is not None:
                    await result

        return CorpusExpansionResult(
            documents_indexed=indexed,
            documents_skipped=skipped,
            chunks_created=chunks_created,
        )
