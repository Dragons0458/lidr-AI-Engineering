from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from streamlit_ui.agent_estimation import (
    HOURS_AGENT_PATH,
    HOURS_DETERMINISTIC_PATH,
    STRUCTURE_AGENT_PATH,
    STRUCTURE_DETERMINISTIC_PATH,
    build_headers,
    calculate_totals,
    estimate_to_rows,
    mark_manual_edits,
    normalize_trace,
    post_agent_hours,
    post_agent_structure,
    post_deterministic_hours,
    post_deterministic_structure,
    post_json,
    restore_session_state,
    rows_to_modules,
    run_to_session_payload,
    serialize_session_state,
    task_hours_to_rows,
    trace_counts,
)


class StubClient:
    def __init__(self, body: Any = None, status: int = 200) -> None:
        self.body = body if body is not None else {"ok": True}
        self.status = status
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})
        request = httpx.Request("POST", url)
        return httpx.Response(self.status, request=request, json=self.body)


@pytest.mark.parametrize(
    ("call", "expected_path", "subject"),
    [
        (post_agent_structure, STRUCTURE_AGENT_PATH, {"project": "Portal"}),
        (post_agent_hours, HOURS_AGENT_PATH, [{"name": "Auth", "tasks": []}]),
        (
            post_deterministic_structure,
            STRUCTURE_DETERMINISTIC_PATH,
            {"project": "Portal"},
        ),
        (
            post_deterministic_hours,
            HOURS_DETERMINISTIC_PATH,
            [{"name": "Auth", "tasks": []}],
        ),
    ],
)
def test_route_helpers_use_expected_paths_and_headers(
    call: Any, expected_path: str, subject: Any
) -> None:
    client = StubClient()
    kwargs: dict[str, Any] = {
        "api_key": " secret ",
        "timeout": 12.5,
        "client": client,
    }
    if call in (post_agent_structure, post_agent_hours):
        call("http://api/", subject, {"model": "gpt-5"}, **kwargs)
    else:
        call("http://api/", subject, **kwargs)

    request = client.calls[0]
    assert request["url"] == f"http://api{expected_path}"
    assert request["headers"] == {"X-API-Key": "secret"}
    assert request["timeout"] == 12.5
    assert build_headers(" ") == {}


def test_post_json_propagates_timeout_and_http_errors() -> None:
    class TimeoutClient:
        @staticmethod
        def post(*args: Any, **kwargs: Any) -> httpx.Response:
            raise httpx.ReadTimeout("slow request")

    with pytest.raises(httpx.ReadTimeout):
        post_json("http://api", "/route", {}, client=TimeoutClient)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        post_json("http://api", "/route", {}, client=StubClient(status=502))
    assert exc_info.value.response.status_code == 502

    with pytest.raises(ValueError, match="JSON object"):
        post_json("http://api", "/route", {}, client=StubClient(body=["invalid"]))


def test_estimate_rows_modules_round_trip_preserves_duplicate_tasks() -> None:
    estimate = {
        "modules": [
            {
                "name": "Auth",
                "description": "Identity",
                "tasks": [
                    {"name": "OAuth", "description": "Provider A"},
                    {"name": "OAuth", "description": "Provider B"},
                ],
            }
        ]
    }
    rows = estimate_to_rows(estimate)
    assert rows == [
        {
            "module": "Auth",
            "module_description": "Identity",
            "task": "OAuth",
            "description": "Provider A",
        },
        {
            "module": "Auth",
            "module_description": "Identity",
            "task": "OAuth",
            "description": "Provider B",
        },
    ]
    modules = rows_to_modules([*rows, {"module": "", "task": "ignored"}])
    assert len(modules) == 1
    assert [task["name"] for task in modules[0]["tasks"]] == ["OAuth", "OAuth"]


def test_task_hours_rows_preserve_provenance_and_default_rate() -> None:
    result = {
        "tasks": [
            {
                "module": "Auth",
                "task": "OAuth",
                "estimated_hours": 8,
                "estimation_source": "agent_recovery",
                "reliability": 0.82,
                "dispersion": 0.12,
                "has_match": True,
                "neighbors": [{"source_id": "chunk-1", "distance": 0.2}],
                "hours_range": {"min": 6, "max": 10},
            }
        ]
    }
    row = task_hours_to_rows(result)[0]
    assert row["hourly_rate_eur"] == 75.0
    assert row["cost_eur"] == 600.0
    assert row["source"] == "agent_recovery"
    assert row["reliability"] == 0.82
    assert row["dispersion"] == 0.12
    assert row["neighbors"][0]["source_id"] == "chunk-1"
    assert row["hours_range"] == {"min": 6, "max": 10}


def test_cost_totals_and_manual_edit_marking() -> None:
    original = [
        {
            "module": "Auth",
            "task": "OAuth",
            "estimated_hours": 8,
            "hourly_rate_eur": 75,
            "source": "deterministic",
        },
        {
            "module": "Core",
            "task": "API",
            "estimated_hours": 4,
            "hourly_rate_eur": 75,
            "source": "agent_recovery",
        },
    ]
    edited = [dict(original[0]), {**original[1], "estimated_hours": 5}]
    marked = mark_manual_edits(original, edited)
    assert marked[0]["source"] == "deterministic"
    assert marked[1]["source"] == "manual"
    assert marked[1]["cost_eur"] == 375.0
    assert calculate_totals(marked) == {
        "total_hours": 13.0,
        "total_engineer_days": 1.62,
        "total_cost_eur": 975.0,
    }


def test_trace_normalization_and_counts() -> None:
    trace = {
        "steps": [
            {"tool": "search_budgets", "reasoning_summary": None},
            {"tool": "derive_task_hours", "reasoning_summary": "Derived."},
        ]
    }
    assert normalize_trace(trace)["steps"][0]["reasoning_summary"].startswith("(sin")
    assert trace_counts(trace) == {
        "steps": 2,
        "search_budgets": 1,
        "derive_task_hours": 1,
    }


def test_session_serialization_restore_and_run_mapping() -> None:
    state = {
        "rag_run_id": 9,
        "rag_payload": {"when": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        "other": "keep",
    }
    serialized = serialize_session_state(state)
    assert serialized["rag_payload"]["when"].startswith("2026-01-01")
    assert "other" not in serialized

    target = {"rag_old": True, "other": "preserved"}
    restore_session_state(target, serialized)
    assert "rag_old" not in target
    assert target["other"] == "preserved"
    assert target["rag_run_id"] == 9

    payload = run_to_session_payload(
        {
            "id": 9,
            "mode": "deterministic",
            "transcript": "Brief",
            "reviewed_structure": [{"module": "Auth"}],
            "final_rows": [{"task": "OAuth"}],
        }
    )
    assert payload["rag_run_id"] == 9
    assert payload["rag_mode"] == "deterministic"
    assert payload["rag_structure_rows"] == [{"module": "Auth"}]
