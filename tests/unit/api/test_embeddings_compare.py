from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.embeddings import router
from app.dependencies import build_chunkers, get_embedder
from app.generation.rag.schemas import Chunk, EmbeddedChunk


class FakeEmbedder:
    def embed_one(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        return [
            EmbeddedChunk(**chunk.model_dump(), embedding=[1.0, 0.0, 0.0])
            for chunk in chunks
        ]


def _sample_budget() -> dict:
    return {
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


def test_compare_endpoint_with_fake_embedder(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    monkeypatch.setattr(
        "app.api.embeddings.build_chunkers",
        lambda names: build_chunkers(["structural"]),
    )
    client = TestClient(app)

    response = client.post(
        "/embeddings/compare",
        json={
            "budgets": [_sample_budget()],
            "queries": ["OAuth authentication"],
            "strategies": ["structural"],
            "top_k": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "structural" in body["stats_per_strategy"]
    assert body["queries_per_strategy"]["structural"][0]["top_k"]


def test_compare_unknown_strategy_returns_400(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()

    def _raise(names):
        raise KeyError(names[0])

    monkeypatch.setattr("app.api.embeddings.build_chunkers", _raise)
    client = TestClient(app)
    response = client.post(
        "/embeddings/compare",
        json={"budgets": [_sample_budget()], "strategies": ["bogus"]},
    )
    assert response.status_code == 400


def test_compare_without_embedder_returns_500():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_embedder] = lambda: None
    client = TestClient(app)
    response = client.post(
        "/embeddings/compare",
        json={"budgets": [_sample_budget()]},
    )
    assert response.status_code == 500
