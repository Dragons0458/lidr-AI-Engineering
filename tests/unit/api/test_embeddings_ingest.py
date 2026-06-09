from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.embeddings import router
from app.dependencies import get_embedder
from app.generation.rag.schemas import Chunk, EmbeddedChunk


class FakeEmbedder:
    model = "text-embedding-3-small"

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        return [
            EmbeddedChunk(**chunk.model_dump(), embedding=[0.1, 0.2, 0.3])
            for chunk in chunks
        ]


def test_ingest_embeddings_endpoint_returns_chunks_and_stats():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    client = TestClient(app)

    response = client.post(
        "/embeddings/ingest",
        json={
            "budgets": [
                {
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
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stats"]["total_budgets"] == 1
    assert body["stats"]["total_chunks"] == 1
    assert body["stats"]["total_tokens"] > 0
    assert body["chunks"][0]["chunk_id"] == "BUD-001::auth"
    assert body["chunks"][0]["embedding"] == [0.1, 0.2, 0.3]
