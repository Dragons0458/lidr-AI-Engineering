"""Liveness and readiness probes (Session 15).

``/health`` is cheap liveness — process is up; never touches the LLM or deps.
``/health/ready`` checks Postgres (``SELECT 1``) and Redis (``PING``) with short
timeouts and returns 503 with per-dependency detail when something is down.
Both stay outside the service-token gate so container HEALTHCHECKs work.
"""

from __future__ import annotations

from datetime import datetime

import redis
import structlog
from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.config import get_settings
from app.foundation.persistence.database import create_engine_from_settings

log = structlog.get_logger()
router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness: the process is alive. Does not call the LLM or databases."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@router.get("/health/ready")
async def health_ready(response: Response) -> dict:
    """Readiness: Postgres and Redis answer. Never calls the LLM."""
    settings = get_settings()
    checks: dict[str, str] = {}

    try:
        engine = create_engine_from_settings()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001 — surface dependency failure
        checks["postgres"] = f"error: {type(exc).__name__}: {exc}"[:200]
        log.warning("health_ready_postgres_failed", error=str(exc)[:200])

    try:
        client = redis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        try:
            if client.ping() is not True:
                raise RuntimeError("PING did not return True")
            checks["redis"] = "ok"
        finally:
            client.close()
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"error: {type(exc).__name__}: {exc}"[:200]
        log.warning("health_ready_redis_failed", error=str(exc)[:200])

    ready = all(value == "ok" for value in checks.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "checks": checks,
            "timestamp": datetime.now().isoformat(),
        }

    return {
        "status": "ready",
        "checks": checks,
        "timestamp": datetime.now().isoformat(),
    }
