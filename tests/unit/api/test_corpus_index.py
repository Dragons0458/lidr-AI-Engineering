"""Tests for corpus index HTTP endpoints (Session 11)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_corpus_index_service
from app.foundation.persistence.database import get_session
from app.generation.rag.index_service import CorpusExpansionResult, CorpusIndexService
from app.generation.rag.schemas import Budget, BudgetComponent, ClientMetadata
from app.main import app


def _budget() -> Budget:
    return Budget(
        budget_id="IDX-001",
        client_metadata=ClientMetadata(name="Acme", sector="ecommerce", country="ES"),
        project_summary="demo",
        main_technology="Rails",
        year=2024,
        total_estimated_hours=10,
        components=[
            BudgetComponent(
                component_id="C1",
                name="Auth",
                description="Login",
                estimated_hours=10,
                complexity="low",
            )
        ],
    )


class _InMemoryJob:
    def __init__(self, source_name: str):
        self.job_id = uuid.uuid4()
        self.source_name = source_name
        self.status = "pending"
        self.documents_count = 0
        self.error_message = None
        self.started_at = datetime.now(timezone.utc)
        self.finished_at = None


class _InMemoryJobsRepo:
    _store: dict[uuid.UUID, _InMemoryJob] = {}

    def __init__(self, session=None) -> None:
        pass

    def create(self, source_name: str):
        job = _InMemoryJob(source_name)
        self._store[job.job_id] = job
        return job

    def get(self, job_id):
        return self._store.get(job_id)

    def mark_running(self, job_id):
        if job := self._store.get(job_id):
            job.status = "running"

    def mark_completed(self, job_id, *, documents_count: int):
        if job := self._store.get(job_id):
            job.status = "completed"
            job.documents_count = documents_count
            job.finished_at = datetime.now(timezone.utc)

    def mark_failed(self, job_id, *, error_message: str):
        if job := self._store.get(job_id):
            job.status = "failed"
            job.error_message = error_message
            job.finished_at = datetime.now(timezone.utc)

    def set_documents_count(self, job_id, count: int):
        if job := self._store.get(job_id):
            job.documents_count = count


class StubService(CorpusIndexService):
    def __init__(self) -> None:
        pass

    async def expand(self, documents, **kwargs):
        return CorpusExpansionResult(
            documents_indexed=len(documents),
            documents_skipped=0,
            chunks_created=len(documents),
        )


@pytest.fixture
def client(monkeypatch) -> TestClient:
    _InMemoryJobsRepo._store.clear()
    service = StubService()
    app.dependency_overrides[get_corpus_index_service] = lambda: service
    app.dependency_overrides[get_session] = lambda: iter([None])
    monkeypatch.setattr(
        "app.api.routers.corpus_index.JobsRepository", _InMemoryJobsRepo
    )

    class _NullSession:
        def close(self):
            pass

    monkeypatch.setattr(
        "app.api.routers.corpus_index.SessionLocal", lambda: _NullSession()
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_post_index_runs_returns_202(client: TestClient) -> None:
    payload = {"documents": [_budget().model_dump()]}
    response = client.post("/embeddings/index/runs", json=payload)
    assert response.status_code == 202
    body = response.json()
    assert body["documents_total"] == 1
    assert body["status"] == "pending"
    uuid.UUID(body["job_id"])


def test_get_index_job_404(client: TestClient) -> None:
    response = client.get(f"/embeddings/index/jobs/{uuid.uuid4()}")
    assert response.status_code == 404


def test_service_none_returns_500() -> None:
    app.dependency_overrides[get_corpus_index_service] = lambda: None
    client = TestClient(app)
    try:
        response = client.post(
            "/embeddings/index/runs",
            json={"documents": [_budget().model_dump()]},
        )
        assert response.status_code == 500
    finally:
        app.dependency_overrides.clear()


def test_get_index_stats_shape(monkeypatch) -> None:
    class StubStore:
        async def corpus_stats(self, session):
            return [("budget", 1, 10, True)]

    class _Session:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *args):
            return None

    def fake_factory():
        return lambda: _Session()

    monkeypatch.setattr(
        "app.api.routers.corpus_index.get_chunk_store", lambda: StubStore()
    )
    monkeypatch.setattr(
        "app.api.routers.corpus_index.get_async_session_factory", fake_factory
    )
    client = TestClient(app)
    response = client.get("/embeddings/index/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["total_chunks"] == 10
    assert body["collections"][0]["hnsw_indexed"] is True
