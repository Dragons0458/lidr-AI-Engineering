"""Helpers for the Session 16 observability Streamlit page (no live server)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from streamlit_ui.common import (
    fetch_observability_metrics,
    fetch_observability_requests,
    load_latest_eval_report,
)


def test_fetch_observability_metrics_ok(monkeypatch) -> None:
    captured: dict = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        request = httpx.Request("GET", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "window_hours": 24,
                "requests": 3,
                "error_rate": 0.0,
                "latency_ms": {"mean": 10.0, "p50": 9.0, "p95": 12.0, "n": 3},
                "cost_usd": {"total": 0.1, "mean_per_request": 0.03, "p95": 0.05},
                "tokens": {"prompt": 1, "completion": 1},
                "abstention_rate": 0.0,
                "cache_hit_rate": 0.0,
                "by_route": [],
                "series": [],
            },
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    payload = fetch_observability_metrics("http://ai-service:8000", 24)
    assert payload["requests"] == 3
    assert captured["url"].endswith("/v1/observability/metrics")
    assert captured["params"]["window_hours"] == 24


def test_fetch_observability_metrics_401(monkeypatch) -> None:
    def fake_get(url, **kwargs):
        request = httpx.Request("GET", url)
        return httpx.Response(401, request=request, json={"detail": "nope"})

    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(httpx.HTTPStatusError):
        fetch_observability_metrics("http://localhost:8000", 1)


def test_fetch_observability_requests(monkeypatch) -> None:
    def fake_get(url, **kwargs):
        request = httpx.Request("GET", url)
        return httpx.Response(
            200,
            request=request,
            json=[{"request_id": "abc", "route": "/v1/estimate/from-transcript"}],
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    rows = fetch_observability_requests("http://localhost:8000", limit=10)
    assert rows[0]["request_id"] == "abc"


def test_load_latest_eval_report(tmp_path: Path) -> None:
    assert load_latest_eval_report(tmp_path) is None
    (tmp_path / "eval_s16_20260101-000000.json").write_text(
        json.dumps({"run_id": "aaa", "arms": {}}), encoding="utf-8"
    )
    (tmp_path / "eval_s16_20260102-000000.json").write_text(
        json.dumps({"run_id": "bbb", "arms": {"rag": {}}}), encoding="utf-8"
    )
    payload = load_latest_eval_report(tmp_path)
    assert payload is not None
    assert payload["run_id"] == "bbb"
