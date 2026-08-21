"""Persist and emit one row of request metrics. Never raises to the caller."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

from app.config import get_settings
from app.foundation.observability.usage import RequestUsage, get_usage

log = structlog.get_logger()

_FORBIDDEN_KEYS = (
    "transcript",
    "prompt",
    "messages",
    "estimation",
    "api_key",
    "token",
    "authorization",
)


@dataclass
class RequestMetrics:
    """One HTTP estimate request. No transcripts, prompts, or secrets."""

    request_id: str
    route: str
    http_status: int
    status: str
    latency_ms: float
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model: str | None = None
    confidence: str | None = None
    abstained: bool = False
    grounded_ratio: float | None = None
    cache_hit: bool = False
    error_type: str | None = None
    created_at: datetime | None = None


def derive_status(
    *,
    http_status: int,
    abstained: bool | None,
) -> str:
    """Map HTTP + abstention onto the stored ``status`` enum."""
    if http_status >= 400:
        return "error"
    if abstained:
        return "abstained"
    return "ok"


def metrics_from_usage(
    *,
    request_id: str,
    route: str,
    http_status: int,
    latency_ms: float,
    usage: RequestUsage | None = None,
    error_type: str | None = None,
) -> RequestMetrics:
    acc = usage if usage is not None else get_usage()
    abstained = bool(acc.outcome_abstained) if acc is not None else False
    models = ",".join(sorted(acc.models)) if acc and acc.models else None
    cache_hit = bool(acc and acc.cache_hits > 0)
    return RequestMetrics(
        request_id=request_id,
        route=route,
        http_status=http_status,
        status=derive_status(http_status=http_status, abstained=abstained),
        latency_ms=float(latency_ms),
        llm_calls=acc.llm_calls if acc else 0,
        prompt_tokens=acc.prompt_tokens if acc else 0,
        completion_tokens=acc.completion_tokens if acc else 0,
        estimated_cost_usd=acc.cost_usd if acc else 0.0,
        model=models,
        confidence=acc.outcome_confidence if acc else None,
        abstained=abstained,
        grounded_ratio=acc.outcome_grounded_ratio if acc else None,
        cache_hit=cache_hit,
        error_type=error_type,
        created_at=datetime.now(timezone.utc),
    )


def _log_payload(metric: RequestMetrics) -> dict[str, Any]:
    payload = {
        "request_id": metric.request_id,
        "route": metric.route,
        "http_status": metric.http_status,
        "status": metric.status,
        "latency_ms": round(metric.latency_ms, 1),
        "llm_calls": metric.llm_calls,
        "prompt_tokens": metric.prompt_tokens,
        "completion_tokens": metric.completion_tokens,
        "estimated_cost_usd": round(metric.estimated_cost_usd, 6),
        "model": metric.model,
        "confidence": metric.confidence,
        "abstained": metric.abstained,
        "grounded_ratio": metric.grounded_ratio,
        "cache_hit": metric.cache_hit,
        "error_type": metric.error_type,
    }
    for key in _FORBIDDEN_KEYS:
        payload.pop(key, None)
    return payload


def _annotate_logfire(metric: RequestMetrics) -> None:
    try:
        import logfire

        logfire.info("estimate_request_metrics", **_log_payload(metric))
    except Exception:  # noqa: BLE001
        return


def record_request_metrics(
    metric: RequestMetrics,
    *,
    repository: Any | None = None,
) -> None:
    """Log + Logfire + INSERT. A metrics failure never breaks the request."""
    try:
        log.info("estimate_request_metrics", **_log_payload(metric))
        _annotate_logfire(metric)
        if repository is not None:
            repository.insert_metric(metric)
            return
        if not get_settings().METRICS_ENABLED:
            return
        from app.foundation.persistence.database import SessionLocal
        from app.foundation.persistence.repositories.metrics import MetricsRepository

        session = SessionLocal()
        try:
            MetricsRepository(session).insert_metric(metric)
        finally:
            session.close()
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "request_metrics_record_failed",
            error_type=type(exc).__name__,
        )
