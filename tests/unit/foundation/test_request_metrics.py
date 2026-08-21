"""record_request_metrics never breaks the caller and never serialises transcripts."""

from __future__ import annotations

from app.foundation.observability.metrics import (
    RequestMetrics,
    derive_status,
    metrics_from_usage,
    record_request_metrics,
)
from app.foundation.observability.usage import add_llm_call, set_outcome, start_usage


def test_derive_status() -> None:
    assert derive_status(http_status=200, abstained=False) == "ok"
    assert derive_status(http_status=200, abstained=True) == "abstained"
    assert derive_status(http_status=502, abstained=False) == "error"
    assert derive_status(http_status=429, abstained=True) == "error"


def test_metrics_from_usage_reads_accumulator() -> None:
    start_usage()
    add_llm_call(
        model="gpt-4o-mini", prompt_tokens=3, completion_tokens=1, cost_usd=0.002
    )
    set_outcome(confidence="insufficient", abstained=True, grounded_ratio=0.0)
    metric = metrics_from_usage(
        request_id="s16-abc-S16-06-1",
        route="/v1/estimate/from-transcript",
        http_status=200,
        latency_ms=1234.5,
    )
    assert metric.llm_calls == 1
    assert metric.estimated_cost_usd == 0.002
    assert metric.status == "abstained"
    assert metric.confidence == "insufficient"
    assert metric.abstained is True


def test_record_request_metrics_swallows_repository_errors() -> None:
    class Boom:
        def insert_metric(self, metric):
            raise RuntimeError("db down")

    metric = RequestMetrics(
        request_id="r1",
        route="/v1/estimate/from-transcript",
        http_status=200,
        status="ok",
        latency_ms=10,
    )
    record_request_metrics(metric, repository=Boom())  # must not raise


def test_payload_has_no_transcript_keys() -> None:
    captured: dict = {}

    class Store:
        def insert_metric(self, metric):
            captured["metric"] = metric

    start_usage()
    add_llm_call(model="gpt-4o-mini", prompt_tokens=1, cost_usd=0.0)
    metric = metrics_from_usage(
        request_id="r2",
        route="/v1/estimate/from-transcript",
        http_status=200,
        latency_ms=1,
    )
    record_request_metrics(metric, repository=Store())
    stored = captured["metric"]
    assert not hasattr(stored, "transcript")
    payload = stored.__dict__
    assert "transcript" not in payload
    assert "prompt" not in payload
    assert "api_key" not in payload
