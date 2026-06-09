from __future__ import annotations

from collections.abc import AsyncIterator
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.embeddings import router
from app.dependencies import get_embedder
from app.foundation.persistence.async_database import get_async_session
from app.generation.rag.schemas import Chunk, EmbeddedChunk

SAMPLE_BUDGET = {
    "budget_id": "BUD-001",
    "client_metadata": {
        "name": "Demo Bank",
        "sector": "finance",
        "country": "Colombia",
    },
    "project_summary": "Banking API modernization",
    "main_technology": "FastAPI",
    "year": 2025,
    "total_estimated_hours": 60,
    "components": [
        {
            "component_id": "auth",
            "name": "Authorization service",
            "description": "JWT access control for banking APIs.",
            "tech_stack": ["FastAPI", "JWT"],
            "estimated_hours": 60,
            "complexity": "high",
            "dependencies": [],
        }
    ],
}

INGEST_PAYLOAD = {
    "source_path": "data/budgets/test_budget.json",
    "document_type": "historical_budget",
    "content": SAMPLE_BUDGET,
}


class FakeEmbedder:
    model = "text-embedding-3-small"

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        return [
            EmbeddedChunk(**chunk.model_dump(), embedding=[0.1] * 1536)
            for chunk in chunks
        ]


class FakeAsyncSession:
    def __init__(self, existing_document_id: int | None = None) -> None:
        self.existing_document_id = existing_document_id
        self.committed = False
        self._next_doc_id = 42

    async def scalar(self, _stmt) -> int | None:
        return self.existing_document_id

    def add(self, obj) -> None:
        if hasattr(obj, "id") and getattr(obj, "id", None) is None:
            obj.id = self._next_doc_id

    async def flush(self) -> None:
        return None

    def add_all(self, _rows) -> None:
        return None

    async def commit(self) -> None:
        self.committed = True


def _build_client(existing_document_id: int | None = None) -> TestClient:
    session = FakeAsyncSession(existing_document_id=existing_document_id)

    async def override_session() -> AsyncIterator[FakeAsyncSession]:
        yield session

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_async_session] = override_session
    return TestClient(app)


def test_ingest_embeddings_endpoint_persists_and_returns_stats():
    client = _build_client()

    response = client.post("/embeddings/ingest", json=INGEST_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == 42
    assert body["chunks_created"] == 1
    assert body["embedding_dimension"] == 1536
    assert body["ingestion_time_ms"] >= 0


def test_ingest_embeddings_endpoint_returns_409_for_duplicate_source_path():
    client = _build_client(existing_document_id=99)

    response = client.post("/embeddings/ingest", json=INGEST_PAYLOAD)

    assert response.status_code == 409
    body = response.json()
    assert body["detail"]["detail"] == "Document already ingested"
    assert body["detail"]["document_id"] == 99
