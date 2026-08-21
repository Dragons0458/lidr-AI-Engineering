"""Session 16 request-metrics repository.

Never leaks ORM types. Callers see ``RequestMetrics`` dataclasses.
Aggregation uses Postgres ``percentile_cont`` so p95 is computed in SQL.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Integer, Select, func, select, text
from sqlalchemy.orm import Session

from app.foundation.observability.metrics import RequestMetrics
from app.foundation.persistence.models import RequestMetricRow


def _to_metric(row: RequestMetricRow) -> RequestMetrics:
    return RequestMetrics(
        request_id=row.request_id,
        route=row.route,
        http_status=row.http_status,
        status=row.status,
        latency_ms=float(row.latency_ms),
        llm_calls=row.llm_calls,
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        estimated_cost_usd=float(row.estimated_cost_usd),
        model=row.model,
        confidence=row.confidence,
        abstained=bool(row.abstained),
        grounded_ratio=(
            float(row.grounded_ratio) if row.grounded_ratio is not None else None
        ),
        cache_hit=bool(row.cache_hit),
        error_type=row.error_type,
        created_at=row.created_at,
    )


class MetricsRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def insert_metric(self, metric: RequestMetrics) -> None:
        row = RequestMetricRow(
            request_id=metric.request_id,
            route=metric.route,
            http_status=metric.http_status,
            status=metric.status,
            latency_ms=metric.latency_ms,
            llm_calls=metric.llm_calls,
            prompt_tokens=metric.prompt_tokens,
            completion_tokens=metric.completion_tokens,
            estimated_cost_usd=metric.estimated_cost_usd,
            model=metric.model,
            confidence=metric.confidence,
            abstained=metric.abstained,
            grounded_ratio=metric.grounded_ratio,
            cache_hit=metric.cache_hit,
            error_type=metric.error_type,
            created_at=metric.created_at or datetime.now(timezone.utc),
        )
        self._session.add(row)
        self._session.commit()

    def list_metrics(
        self,
        *,
        since: datetime | None = None,
        route: str | None = None,
        request_id_prefix: str | None = None,
        limit: int = 50,
    ) -> list[RequestMetrics]:
        stmt: Select[Any] = select(RequestMetricRow).order_by(
            RequestMetricRow.created_at.desc()
        )
        if since is not None:
            stmt = stmt.where(RequestMetricRow.created_at >= since)
        if route:
            stmt = stmt.where(RequestMetricRow.route == route)
        if request_id_prefix:
            stmt = stmt.where(RequestMetricRow.request_id.like(f"{request_id_prefix}%"))
        stmt = stmt.limit(max(1, min(limit, 500)))
        rows = self._session.scalars(stmt).all()
        return [_to_metric(row) for row in rows]

    def aggregate_metrics(
        self,
        *,
        window_hours: int = 24,
        route: str | None = None,
        bucket_minutes: int = 60,
    ) -> dict[str, Any]:
        since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        filters = [RequestMetricRow.created_at >= since]
        if route:
            filters.append(RequestMetricRow.route == route)

        totals = self._session.execute(
            select(
                func.count(RequestMetricRow.id),
                func.avg(RequestMetricRow.latency_ms),
                func.percentile_cont(0.5).within_group(RequestMetricRow.latency_ms),
                func.percentile_cont(0.95).within_group(RequestMetricRow.latency_ms),
                func.avg(RequestMetricRow.estimated_cost_usd),
                func.percentile_cont(0.95).within_group(
                    RequestMetricRow.estimated_cost_usd
                ),
                func.coalesce(func.sum(RequestMetricRow.estimated_cost_usd), 0),
                func.coalesce(func.sum(RequestMetricRow.prompt_tokens), 0),
                func.coalesce(func.sum(RequestMetricRow.completion_tokens), 0),
                func.coalesce(
                    func.sum(func.cast(RequestMetricRow.abstained, Integer)),
                    0,
                ),
                func.coalesce(
                    func.sum(func.cast(RequestMetricRow.cache_hit, Integer)),
                    0,
                ),
                func.coalesce(
                    func.sum(func.cast(RequestMetricRow.status == "error", Integer)),
                    0,
                ),
            ).where(*filters)
        ).one()

        n = int(totals[0] or 0)
        empty = {
            "window_hours": window_hours,
            "requests": 0,
            "error_rate": 0.0,
            "latency_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "n": 0},
            "cost_usd": {"total": 0.0, "mean_per_request": 0.0, "p95": 0.0},
            "tokens": {"prompt": 0, "completion": 0},
            "abstention_rate": 0.0,
            "cache_hit_rate": 0.0,
            "by_route": [],
            "series": [],
        }
        if n == 0:
            return empty

        error_count = int(totals[11] or 0)
        abstained_count = int(totals[9] or 0)
        cache_hits = int(totals[10] or 0)
        summary = {
            "window_hours": window_hours,
            "requests": n,
            "error_rate": error_count / n,
            "latency_ms": {
                "mean": float(totals[1] or 0.0),
                "p50": float(totals[2] or 0.0),
                "p95": float(totals[3] or 0.0),
                "n": n,
            },
            "cost_usd": {
                "total": float(totals[6] or 0.0),
                "mean_per_request": float(totals[4] or 0.0),
                "p95": float(totals[5] or 0.0),
            },
            "tokens": {
                "prompt": int(totals[7] or 0),
                "completion": int(totals[8] or 0),
            },
            "abstention_rate": abstained_count / n,
            "cache_hit_rate": cache_hits / n,
            "by_route": self._by_route(filters),
            "series": self._series(filters, bucket_minutes),
        }
        return summary

    def _by_route(self, filters: list) -> list[dict[str, Any]]:
        rows = self._session.execute(
            select(
                RequestMetricRow.route,
                func.count(RequestMetricRow.id),
                func.avg(RequestMetricRow.latency_ms),
                func.percentile_cont(0.95).within_group(RequestMetricRow.latency_ms),
                func.coalesce(func.sum(RequestMetricRow.estimated_cost_usd), 0),
                func.coalesce(
                    func.sum(func.cast(RequestMetricRow.status == "error", Integer)),
                    0,
                ),
            )
            .where(*filters)
            .group_by(RequestMetricRow.route)
            .order_by(func.count(RequestMetricRow.id).desc())
        ).all()
        result = []
        for row in rows:
            count = int(row[1] or 0)
            result.append(
                {
                    "route": row[0],
                    "requests": count,
                    "mean_latency_ms": float(row[2] or 0.0),
                    "p95_latency_ms": float(row[3] or 0.0),
                    "cost_usd": float(row[4] or 0.0),
                    "error_rate": (int(row[5] or 0) / count) if count else 0.0,
                }
            )
        return result

    def _series(self, filters: list, bucket_minutes: int) -> list[dict[str, Any]]:
        minutes = max(1, bucket_minutes)
        bucket = func.to_timestamp(
            func.floor(
                func.extract("epoch", RequestMetricRow.created_at) / (minutes * 60)
            )
            * (minutes * 60)
        )
        rows = self._session.execute(
            select(
                bucket.label("bucket"),
                func.count(RequestMetricRow.id),
                func.percentile_cont(0.95).within_group(RequestMetricRow.latency_ms),
                func.coalesce(func.sum(RequestMetricRow.estimated_cost_usd), 0),
                func.coalesce(
                    func.sum(func.cast(RequestMetricRow.status == "error", Integer)),
                    0,
                ),
            )
            .where(*filters)
            .group_by(text("bucket"))
            .order_by(text("bucket"))
        ).all()
        series = []
        for row in rows:
            count = int(row[1] or 0)
            bucket_dt = row[0]
            series.append(
                {
                    "bucket": bucket_dt.isoformat() if bucket_dt is not None else None,
                    "requests": count,
                    "p95_latency_ms": float(row[2] or 0.0),
                    "cost_usd": float(row[3] or 0.0),
                    "error_rate": (int(row[4] or 0) / count) if count else 0.0,
                }
            )
        return series
