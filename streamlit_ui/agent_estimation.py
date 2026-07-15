"""Pure HTTP, transformation, and state helpers for the hybrid RAG wizard."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, MutableMapping
from typing import Any

import httpx

from streamlit_ui.agents import STREAMLIT_DEFAULT_HOURLY_RATE_EUR

STRUCTURE_AGENT_PATH = "/v1/estimate/agent/structure"
HOURS_AGENT_PATH = "/v1/estimate/agent/hours"
REFORMULATE_PATH = "/v1/estimate/stages/reformulate"
STRUCTURE_DETERMINISTIC_PATH = "/v1/estimate/stages/structure"
HOURS_DETERMINISTIC_PATH = "/v1/estimate/tasks/hours"
ONE_SHOT_PATH = "/v1/estimate/from-transcript"
DEFAULT_HTTP_TIMEOUT = 600.0


def build_headers(api_key: str | None) -> dict[str, str]:
    return {"X-API-Key": api_key.strip()} if api_key and api_key.strip() else {}


def post_json(
    api_root: str,
    path: str,
    payload: Mapping[str, Any],
    *,
    api_key: str | None = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    client: Any = httpx,
) -> dict[str, Any]:
    """POST JSON and return a validated object; HTTP errors remain inspectable."""
    response = client.post(
        f"{api_root.rstrip('/')}/{path.lstrip('/')}",
        json=dict(payload),
        headers=build_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("The API response must be a JSON object.")
    return body


def post_agent_structure(
    api_root: str,
    query: Mapping[str, Any],
    profile_payload: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return post_json(
        api_root,
        STRUCTURE_AGENT_PATH,
        {"query": dict(query), **dict(profile_payload or {})},
        **kwargs,
    )


def post_agent_hours(
    api_root: str,
    modules: list[dict[str, Any]],
    profile_payload: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return post_json(
        api_root,
        HOURS_AGENT_PATH,
        {"modules": modules, **dict(profile_payload or {})},
        **kwargs,
    )


def post_deterministic_structure(
    api_root: str, query: Mapping[str, Any], **kwargs: Any
) -> dict[str, Any]:
    return post_json(
        api_root, STRUCTURE_DETERMINISTIC_PATH, {"query": dict(query)}, **kwargs
    )


def post_deterministic_hours(
    api_root: str, modules: list[dict[str, Any]], **kwargs: Any
) -> dict[str, Any]:
    return post_json(api_root, HOURS_DETERMINISTIC_PATH, {"modules": modules}, **kwargs)


def estimate_to_rows(estimate: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for module in (estimate or {}).get("modules") or []:
        for task in module.get("tasks") or []:
            rows.append(
                {
                    "module": str(module.get("name") or ""),
                    "module_description": str(module.get("description") or ""),
                    "task": str(task.get("name") or ""),
                    "description": str(task.get("description") or ""),
                }
            )
    return rows


def rows_to_modules(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group reviewed rows without collapsing repeated tasks inside a module."""
    modules: dict[str, dict[str, Any]] = {}
    for row in rows:
        module_name = str(row.get("module") or "").strip()
        task_name = str(row.get("task") or "").strip()
        if not module_name or not task_name:
            continue
        module = modules.setdefault(
            module_name,
            {
                "name": module_name,
                "description": str(row.get("module_description") or "").strip() or None,
                "tasks": [],
            },
        )
        module["tasks"].append(
            {
                "name": task_name,
                "description": str(row.get("description") or "").strip() or None,
            }
        )
    return list(modules.values())


def task_hours_to_rows(
    result: Mapping[str, Any] | None,
    *,
    hourly_rate_eur: float = STREAMLIT_DEFAULT_HOURLY_RATE_EUR,
) -> list[dict[str, Any]]:
    rate = hourly_rate_eur if hourly_rate_eur > 0 else STREAMLIT_DEFAULT_HOURLY_RATE_EUR
    rows: list[dict[str, Any]] = []
    for task in (result or {}).get("tasks") or []:
        hours = task.get("estimated_hours")
        row = {
            "module": task.get("module"),
            "task": task.get("task"),
            "estimated_hours": hours,
            "hourly_rate_eur": rate,
            "cost_eur": row_cost(hours, rate),
            "source": task.get("estimation_source") or "deterministic",
            "reliability": task.get("reliability"),
            "dispersion": task.get("dispersion"),
            "has_match": bool(task.get("has_match")),
            "neighbors": copy.deepcopy(task.get("neighbors") or []),
            "hours_range": copy.deepcopy(task.get("hours_range")),
        }
        rows.append(row)
    return rows


def normalize_trace(trace: Mapping[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    steps: list[dict[str, Any]] = []
    for index, raw in enumerate((trace or {}).get("steps") or [], start=1):
        steps.append(
            {
                "step": int(raw.get("step") or index),
                "reasoning_summary": raw.get("reasoning_summary")
                or "(sin resumen de razonamiento)",
                "tool": str(raw.get("tool") or ""),
                "tool_args": copy.deepcopy(raw.get("tool_args") or {}),
                "observation": str(raw.get("observation") or ""),
            }
        )
    return {"steps": steps}


def trace_counts(trace: Mapping[str, Any] | None) -> dict[str, int]:
    steps = normalize_trace(trace)["steps"]
    return {
        "steps": len(steps),
        "search_budgets": sum(s["tool"] == "search_budgets" for s in steps),
        "derive_task_hours": sum(s["tool"] == "derive_task_hours" for s in steps),
    }


def row_cost(hours: Any, hourly_rate_eur: Any = None) -> float:
    rate = (
        STREAMLIT_DEFAULT_HOURLY_RATE_EUR
        if hourly_rate_eur in (None, "")
        else float(hourly_rate_eur)
    )
    if rate <= 0:
        rate = STREAMLIT_DEFAULT_HOURLY_RATE_EUR
    return round(float(hours or 0) * rate, 2)


def calculate_totals(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    total_hours = round(sum(float(row.get("estimated_hours") or 0) for row in rows), 2)
    total_cost = round(
        sum(
            row_cost(row.get("estimated_hours"), row.get("hourly_rate_eur"))
            for row in rows
        ),
        2,
    )
    return {
        "total_hours": total_hours,
        "total_engineer_days": round(total_hours / 8, 2),
        "total_cost_eur": total_cost,
    }


def mark_manual_edits(
    original_rows: list[Mapping[str, Any]],
    edited_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Mark rows manual only when hours or rate differ from the generated value."""
    output: list[dict[str, Any]] = []
    for index, edited in enumerate(edited_rows):
        row = dict(edited)
        original = original_rows[index] if index < len(original_rows) else {}
        if row.get("estimated_hours") != original.get("estimated_hours") or row.get(
            "hourly_rate_eur"
        ) != original.get("hourly_rate_eur"):
            row["source"] = "manual"
        row["cost_eur"] = row_cost(
            row.get("estimated_hours"), row.get("hourly_rate_eur")
        )
        output.append(row)
    return output


def serialize_session_state(
    state: Mapping[str, Any], *, prefix: str = "rag_"
) -> dict[str, Any]:
    """Return a JSON-safe copy of wizard-owned session values."""
    selected = {key: value for key, value in state.items() if key.startswith(prefix)}
    return json.loads(json.dumps(selected, ensure_ascii=False, default=str))


def restore_session_state(
    state: MutableMapping[str, Any],
    payload: Mapping[str, Any],
    *,
    clear_prefix: str = "rag_",
) -> None:
    for key in [key for key in state if key.startswith(clear_prefix)]:
        del state[key]
    for key, value in payload.items():
        if key.startswith(clear_prefix):
            state[key] = copy.deepcopy(value)


def run_to_session_payload(run: Mapping[str, Any]) -> dict[str, Any]:
    """Map a persisted run to the wizard's stable session-state contract."""
    return {
        "rag_run_id": run.get("id"),
        "rag_mode": run.get("mode", "agentic"),
        "rag_transcript": run.get("transcript", ""),
        "rag_reformulation": run.get("reformulation_payload"),
        "rag_structure": run.get("structure_response"),
        "rag_structure_rows": run.get("reviewed_structure"),
        "rag_hours": run.get("task_hours_response"),
        "rag_gate_report": run.get("gate_report"),
        "rag_final_rows": run.get("final_rows"),
        "rag_structure_profile_id": run.get("structure_profile_id"),
        "rag_hours_profile_id": run.get("hours_profile_id"),
    }


structure_rows_from_estimate = estimate_to_rows
modules_payload_from_rows = rows_to_modules
final_rows_from_task_hours = task_hours_to_rows
calculate_estimation_totals = calculate_totals
