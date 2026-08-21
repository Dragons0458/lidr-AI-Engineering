"""``GET /v1/observability/metrics`` and ``/requests`` — Session 16 dashboard feed."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from app.api.rate_limiting import limiter
from app.api.security import require_estimate_key
from app.domain.schemas.observability import MetricsSummary, RequestMetricView
from app.foundation.persistence.database import get_session
from app.foundation.persistence.repositories.metrics import MetricsRepository

router = APIRouter(prefix="/v1/observability", tags=["observability"])


def get_metrics_repo(session: Session = Depends(get_session)) -> MetricsRepository:
    return MetricsRepository(session)


@router.get(
    "/metrics",
    response_model=MetricsSummary,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("60/minute")
def observability_metrics(
    request: Request,
    window_hours: int = Query(default=24, ge=1, le=168),
    route: str | None = Query(default=None),
    bucket_minutes: int = Query(default=60, ge=1, le=1440),
    repo: MetricsRepository = Depends(get_metrics_repo),
) -> MetricsSummary:
    """Aggregated vital signs for the selected window. Empty window → zeros."""
    payload = repo.aggregate_metrics(
        window_hours=window_hours,
        route=route,
        bucket_minutes=bucket_minutes,
    )
    return MetricsSummary.model_validate(payload)


@router.get(
    "/requests",
    response_model=list[RequestMetricView],
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("60/minute")
def observability_requests(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    request_id_prefix: str | None = Query(default=None),
    since: datetime | None = Query(default=None),
    route: str | None = Query(default=None),
    repo: MetricsRepository = Depends(get_metrics_repo),
) -> list[RequestMetricView]:
    """Recent instrumented requests. Used by the harness to join real cost."""
    rows = repo.list_metrics(
        since=since,
        route=route,
        request_id_prefix=request_id_prefix,
        limit=limit,
    )
    return [RequestMetricView.model_validate(row, from_attributes=True) for row in rows]
