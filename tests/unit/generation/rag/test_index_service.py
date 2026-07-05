"""Unit tests for CorpusIndexService (Session 11)."""

from __future__ import annotations

import pytest

from app.generation.rag.index_service import CorpusIndexService
from app.generation.rag.ingest_service import DuplicateDocumentError
from app.generation.rag.schemas import (
    Budget,
    BudgetComponent,
    ClientMetadata,
    IngestResponse,
)


def _budget(bid: str) -> Budget:
    return Budget(
        budget_id=bid,
        client_metadata=ClientMetadata(name="Acme", sector="ecommerce", country="ES"),
        project_summary="demo",
        main_technology="Rails",
        year=2024,
        total_estimated_hours=40,
        components=[
            BudgetComponent(
                component_id="C1",
                name="Auth",
                description="Login",
                estimated_hours=40,
                complexity="low",
            )
        ],
    )


class FakeIngest:
    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.calls = 0

    async def ingest(
        self, *, source_path: str, document_type: str, budget: Budget, chunk_type: str
    ):
        self.calls += 1
        if budget.budget_id in self.seen:
            raise DuplicateDocumentError(99)
        self.seen.add(budget.budget_id)
        return IngestResponse(
            document_id=len(self.seen),
            chunks_created=1,
            embedding_dimension=1536,
            ingestion_time_ms=1,
        )


@pytest.mark.asyncio
async def test_expand_indexes_new_and_skips_duplicates():
    ingest = FakeIngest()
    service = CorpusIndexService(ingest=ingest)  # type: ignore[arg-type]
    docs = [_budget("B-1"), _budget("B-1"), _budget("B-2")]
    progress: list[int] = []

    async def on_progress(n: int) -> None:
        progress.append(n)

    result = await service.expand(
        docs,
        document_type="historical_budget",
        chunk_type="budget_component",
        on_progress=on_progress,
    )
    assert result.documents_indexed == 2
    assert result.documents_skipped == 1
    assert result.chunks_created == 2
    assert progress == [1, 2, 3]
