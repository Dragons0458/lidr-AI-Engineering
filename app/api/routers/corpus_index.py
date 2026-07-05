"""``POST/GET /embeddings/index/*`` — incremental corpus expansion (Session 11)."""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.dependencies import get_chunk_store, get_corpus_index_service
from app.foundation.persistence.async_database import get_async_session_factory
from app.foundation.persistence.database import SessionLocal, get_session
from app.foundation.persistence.repositories.jobs import JobsRepository
from app.generation.rag.index_service import CorpusIndexService
from app.generation.rag.schemas import (
    CollectionStats,
    CorpusStats,
    IndexJobView,
    IndexRunRequest,
    IndexRunResponse,
)

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["corpus-index"])


def _job_update(job_id: uuid.UUID, fn) -> None:
    session = SessionLocal()
    try:
        fn(JobsRepository(session))
    finally:
        session.close()


async def _run_expansion(
    *,
    job_id: uuid.UUID,
    request: IndexRunRequest,
    service: CorpusIndexService,
) -> None:
    def mark_running(repo: JobsRepository) -> None:
        repo.mark_running(job_id)

    _job_update(job_id, mark_running)

    def on_progress(count: int) -> None:
        def _set_count(repo: JobsRepository) -> None:
            repo.set_documents_count(job_id, count)

        _job_update(job_id, _set_count)

    try:
        result = await service.expand(
            request.documents,
            document_type=request.document_type,
            chunk_type=request.chunk_type,
            on_progress=on_progress,
        )

        def mark_completed(repo: JobsRepository) -> None:
            repo.mark_completed(
                job_id,
                documents_count=result.documents_indexed + result.documents_skipped,
            )

        _job_update(job_id, mark_completed)
        log.info(
            "corpus_expansion_completed",
            job_id=str(job_id),
            indexed=result.documents_indexed,
            skipped=result.documents_skipped,
            chunks_created=result.chunks_created,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("corpus_expansion_failed", job_id=str(job_id), error=str(exc)[:400])
        error_message = str(exc)

        def mark_failed(repo: JobsRepository) -> None:
            repo.mark_failed(job_id, error_message=error_message)

        _job_update(job_id, mark_failed)


@router.post("/index/runs", response_model=IndexRunResponse, status_code=202)
def create_index_run(
    request: IndexRunRequest,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
    service: CorpusIndexService | None = Depends(get_corpus_index_service),
) -> IndexRunResponse:
    if service is None:
        raise HTTPException(
            status_code=500, detail="Corpus index service is not available."
        )

    repo = JobsRepository(session)
    job = repo.create(source_name="corpus-expansion")
    background.add_task(
        _run_expansion,
        job_id=job.job_id,
        request=request,
        service=service,
    )
    return IndexRunResponse(
        job_id=job.job_id,
        documents_total=len(request.documents),
        status=job.status,
    )


@router.get("/index/jobs/{job_id}", response_model=IndexJobView)
def get_index_job(
    job_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> IndexJobView:
    job = JobsRepository(session).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return IndexJobView(
        job_id=job.job_id,
        status=job.status,
        documents_processed=job.documents_count,
        error_message=job.error_message,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get("/index/stats", response_model=CorpusStats)
async def get_corpus_stats() -> CorpusStats:
    store = get_chunk_store()
    session_factory = get_async_session_factory()
    async with session_factory() as session:
        rows = await store.corpus_stats(session)

    collections = [
        CollectionStats(
            collection=name,
            documents=docs,
            chunks=chunks,
            hnsw_indexed=hnsw,
        )
        for name, docs, chunks, hnsw in rows
    ]
    return CorpusStats(
        collections=collections,
        total_chunks=sum(c.chunks for c in collections),
    )
