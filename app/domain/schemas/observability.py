"""HTTP contracts for Session 16 observability endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class LatencyStats(BaseModel):
    mean: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    n: int = 0


class CostStats(BaseModel):
    total: float = 0.0
    mean_per_request: float = 0.0
    p95: float = 0.0


class TokenStats(BaseModel):
    prompt: int = 0
    completion: int = 0


class RouteBreakdown(BaseModel):
    route: str
    requests: int = 0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    cost_usd: float = 0.0
    error_rate: float = 0.0


class SeriesPoint(BaseModel):
    bucket: datetime | None = None
    requests: int = 0
    p95_latency_ms: float = 0.0
    cost_usd: float = 0.0
    error_rate: float = 0.0


class MetricsSummary(BaseModel):
    window_hours: int
    requests: int = 0
    error_rate: float = 0.0
    latency_ms: LatencyStats = Field(default_factory=LatencyStats)
    cost_usd: CostStats = Field(default_factory=CostStats)
    tokens: TokenStats = Field(default_factory=TokenStats)
    abstention_rate: float = 0.0
    cache_hit_rate: float = 0.0
    by_route: list[RouteBreakdown] = Field(default_factory=list)
    series: list[SeriesPoint] = Field(default_factory=list)


class RequestMetricView(BaseModel):
    request_id: str
    route: str
    http_status: int
    status: str
    latency_ms: float
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    model: str | None = None
    confidence: str | None = None
    abstained: bool = False
    grounded_ratio: float | None = None
    cache_hit: bool = False
    error_type: str | None = None
    created_at: datetime | None = None
