"""Normalise RAG / graph HTTP payloads onto a common Outcome (engineer-days)."""

from __future__ import annotations

from typing import Any

from evals.production.schemas import Outcome

_BUDGET_ID_PREFIX = "S07-"


def _as_dict(payload: Any) -> dict:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    dump = getattr(payload, "model_dump", None)
    if callable(dump):
        return dump()
    return dict(payload)


def _looks_like_budget_id(value: object) -> bool:
    text = str(value).strip()
    return text.startswith(_BUDGET_ID_PREFIX) or text.startswith("S")


def _collect_source_ids_from_rag(body: dict) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        if value is None:
            return
        text = str(value).strip()
        if not text or text in seen:
            return
        # document_id is the parent budget id on SourceReference; skip raw ints
        # unless they already look like a corpus id.
        if text.isdigit() and not _looks_like_budget_id(text):
            return
        seen.add(text)
        found.append(text)

    for module in body.get("modules") or []:
        for task in module.get("tasks") or []:
            for source in task.get("sources") or []:
                if isinstance(source, dict):
                    add(source.get("document_id"))
    for source in body.get("sources") or []:
        if isinstance(source, dict):
            add(source.get("document_id") or source.get("source_id"))
    return found


def normalize_rag_response(
    body: dict | None,
    *,
    latency_ms: float = 0.0,
    http_status: int | None = None,
    error: str | None = None,
    throttled: bool = False,
) -> Outcome:
    """Map ``POST /v1/estimate/from-transcript`` onto ``Outcome``."""
    if throttled:
        return Outcome(
            latency_ms=latency_ms,
            http_status=http_status,
            throttled=True,
            error=error or "throttled",
            abstention_signal="explicit",
            llm_calls=1,
        )
    if error and not body:
        return Outcome(
            latency_ms=latency_ms,
            http_status=http_status,
            error=error,
            llm_calls=1,
            abstention_signal="explicit",
        )

    payload = _as_dict(body)
    confidence = payload.get("confidence")
    days = payload.get("total_engineer_days")
    if days is not None:
        days = int(days)
    abstained = confidence == "insufficient" or days is None
    assumptions = payload.get("assumptions") or []
    return Outcome(
        engineer_days=None if abstained else days,
        confidence=confidence,
        abstained=bool(abstained),
        abstention_signal="explicit",
        source_ids=_collect_source_ids_from_rag(payload),
        assumptions_count=len(assumptions),
        latency_ms=latency_ms,
        llm_calls=1,
        http_status=http_status,
        error=error,
    )


def _collect_source_ids_from_graph(state: dict) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        if not value:
            return
        text = str(value).strip()
        if not text or text in seen:
            return
        seen.add(text)
        found.append(text)

    for row in state.get("task_hours") or []:
        if not isinstance(row, dict):
            continue
        add(row.get("reference_budget_id") or row.get("budget_id"))
        for neighbor in row.get("neighbors") or []:
            if isinstance(neighbor, dict):
                add(neighbor.get("budget_id"))

    estimate = state.get("estimate") or {}
    if isinstance(estimate, dict):
        for module in estimate.get("modules") or []:
            for task in module.get("tasks") or []:
                if not isinstance(task, dict):
                    continue
                add(task.get("reference_budget_id") or task.get("budget_id"))
                for neighbor in task.get("neighbors") or []:
                    if isinstance(neighbor, dict):
                        add(neighbor.get("budget_id"))
    return found


def _graph_tasks(estimate: dict, task_hours: list) -> list[dict]:
    tasks: list[dict] = []
    for module in estimate.get("modules") or []:
        for task in module.get("tasks") or []:
            if isinstance(task, dict):
                tasks.append(task)
    if tasks:
        return tasks
    return [row for row in task_hours if isinstance(row, dict)]


def _graph_abstained(estimate: dict | None, task_hours: list) -> bool:
    """Proxy for 'I don't know': the graph has no explicit insufficient path."""
    if not estimate:
        return False
    ratio = estimate.get("grounded_task_ratio")
    if ratio == 0.0:
        return True
    tasks = _graph_tasks(estimate, task_hours)
    has_match = any(bool(task.get("has_match")) for task in tasks)
    return estimate.get("confidence") == "low" and not has_match


def normalize_graph_state(
    state: dict | None,
    *,
    latency_ms: float = 0.0,
    http_status: int | None = None,
    error: str | None = None,
    throttled: bool = False,
    skipped: bool = False,
    skip_reason: str | None = None,
    llm_calls: int = 0,
) -> Outcome:
    """Map ``GET /v1/estimate/agent/graph/{id}/state`` onto ``Outcome``."""
    if skipped:
        return Outcome(
            latency_ms=latency_ms,
            http_status=http_status,
            skipped=True,
            skip_reason=skip_reason or error or "skipped",
            error=error,
            abstention_signal="proxy",
            llm_calls=llm_calls,
        )
    if throttled:
        return Outcome(
            latency_ms=latency_ms,
            http_status=http_status,
            throttled=True,
            error=error or "throttled",
            abstention_signal="proxy",
            llm_calls=llm_calls,
        )
    if error and not state:
        return Outcome(
            latency_ms=latency_ms,
            http_status=http_status,
            error=error,
            llm_calls=llm_calls,
            abstention_signal="proxy",
        )

    payload = _as_dict(state)
    estimate = (
        payload.get("estimate") if isinstance(payload.get("estimate"), dict) else {}
    )
    task_hours = payload.get("task_hours") or []
    days = estimate.get("total_engineer_days") if estimate else None
    if days is not None:
        days = int(round(float(days)))
    ratio = estimate.get("grounded_task_ratio") if estimate else None
    abstained = _graph_abstained(estimate or None, task_hours)
    assumptions = estimate.get("assumptions") if estimate else None
    if not isinstance(assumptions, list):
        assumptions = []
    return Outcome(
        engineer_days=None if abstained else days,
        confidence=estimate.get("confidence") if estimate else None,
        abstained=abstained,
        abstention_signal="proxy",
        source_ids=_collect_source_ids_from_graph(payload),
        assumptions_count=len(assumptions),
        grounded_ratio=float(ratio) if ratio is not None else None,
        latency_ms=latency_ms,
        llm_calls=llm_calls,
        http_status=http_status,
        error=error,
    )
