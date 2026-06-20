"""Per-stage structured logging for the RAG pipeline (Session 9)."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

import structlog

log = structlog.get_logger()


@contextmanager
def log_stage(stage: str, request_id: str, **context: Any) -> Iterator[None]:
    """Log the lifecycle of one pipeline stage bound to ``request_id``."""
    log.info("stage.started", stage=stage, request_id=request_id, **context)
    t0 = time.perf_counter()
    try:
        yield
    except Exception as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        log.error(
            "stage.failed",
            stage=stage,
            request_id=request_id,
            duration_ms=duration_ms,
            error_type=type(exc).__name__,
            error=str(exc)[:300],
            **context,
        )
        raise
    duration_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        "stage.completed",
        stage=stage,
        request_id=request_id,
        duration_ms=duration_ms,
        **context,
    )
