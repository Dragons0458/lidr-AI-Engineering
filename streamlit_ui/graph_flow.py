"""Streamlit helpers for the Session 13 multi-agent graph wizard."""

from __future__ import annotations

import io
from typing import Any

import httpx

from streamlit_ui.common import get_api_root_url, get_estimate_api_key

GRAPH_PREFIX = "/v1/estimate/agent/graph"

GRAPH_NODES: list[dict[str, Any]] = [
    {
        "key": "classifier",
        "label": "Classifier",
        "node_fn": "classifier_agent",
        "kind": "agent",
        "model": "gpt-5-mini",
        "config_key": "GRAPH_CLASSIFIER_MODEL",
        "role": "Clasifica la complejidad y reformula el brief.",
        "explanation": (
            "Lee la transcripción cruda y produce complexity + brief reformulado. "
            "Handover explícito (Command) hacia structure_agent."
        ),
        "edge": "handover",
        "character": {"name": "Morpheus", "avatar": "🕶"},
    },
    {
        "key": "structure",
        "label": "Structure",
        "node_fn": "structure_agent",
        "kind": "agent",
        "model": "gpt-5",
        "config_key": "AGENT_MODEL",
        "role": "Descompone el brief en módulos → tareas (sin horas).",
        "explanation": "Reutiliza el agente S12 (Responses API) con effort según complejidad.",
        "edge": "edge",
        "character": {"name": "Neo", "avatar": "🥋"},
    },
    {
        "key": "gate_structure",
        "label": "🧑 Gate 1 · revisión de estructura",
        "node_fn": "human_gate_structure",
        "kind": "gate",
        "model": None,
        "config_key": None,
        "role": "Pausa para revisión humana de módulos y tareas.",
        "explanation": "interrupt() primero; al reanudar devuelve approved_modules.",
        "edge": "send",
        "character": {"name": "El Operador", "avatar": "🧑"},
    },
    {
        "key": "hours",
        "label": "Hours ×N",
        "node_fn": "estimate_task_hours",
        "kind": "fanout",
        "model": None,
        "config_key": "TASK_HOURS_TOP_K / TASK_HOURS_DISTANCE_THRESHOLD",
        "role": "Horas por tarea vía búsqueda vectorial determinista.",
        "explanation": "Fan-out Send: una rama por tarea aprobada, sin LLM.",
        "edge": "join",
        "character": {"name": "Tank", "avatar": "🎛"},
    },
    {
        "key": "recover",
        "label": "Recover & handover",
        "node_fn": "recover_and_handover",
        "kind": "join",
        "model": "gpt-5",
        "config_key": "AGENT_MODEL",
        "role": "Junta ramas, recupera tareas dudosas y construye la estimación.",
        "explanation": "Join del fan-out + recovery agéntico opcional. Handover a analysis.",
        "edge": "handover",
        "character": {"name": "Trinity", "avatar": "🏍"},
    },
    {
        "key": "analysis",
        "label": "Analysis",
        "node_fn": "analysis_agent",
        "kind": "agent",
        "model": "gpt-5",
        "config_key": "GRAPH_ANALYSIS_MODEL",
        "role": "Redacta el informe de fiabilidad (no toca los números).",
        "explanation": "ReliabilityReport con ratio determinista sobrescrito en Python.",
        "edge": "edge",
        "character": {"name": "El Oráculo", "avatar": "🔮"},
    },
    {
        "key": "gate_analysis",
        "label": "🧑 Gate 2 · validación final",
        "node_fn": "human_gate_analysis",
        "kind": "gate",
        "model": None,
        "config_key": None,
        "role": "Pausa para validación humana del informe y la estimación.",
        "explanation": "interrupt() + estimate_overrides + want_proposal.",
        "edge": "conditional",
        "character": {"name": "El Operador", "avatar": "🧑"},
    },
    {
        "key": "proposal",
        "label": "Proposal",
        "node_fn": "proposal_agent",
        "kind": "agent",
        "model": "gpt-5",
        "config_key": "GRAPH_PROPOSAL_MODEL",
        "role": "Bonus: redacta la propuesta comercial.",
        "explanation": "Solo si GRAPH_PROPOSAL_ENABLED y want_proposal en gate 2.",
        "edge": "end",
        "character": {"name": "El Arquitecto", "avatar": "📐"},
    },
]


def _headers(api_key: str | None = None) -> dict[str, str]:
    key = (api_key or get_estimate_api_key() or "").strip()
    return {"X-API-Key": key, "Content-Type": "application/json"}


def graph_start_stream(
    transcript: str,
    estimation_id: str,
    *,
    api_root: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    root = (api_root or get_api_root_url()).rstrip("/")
    response = httpx.post(
        f"{root}{GRAPH_PREFIX}/stream",
        json={"estimation_id": estimation_id, "transcript": transcript},
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def graph_resume_stream(
    estimation_id: str,
    decision: dict[str, Any],
    *,
    api_root: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    root = (api_root or get_api_root_url()).rstrip("/")
    response = httpx.post(
        f"{root}{GRAPH_PREFIX}/{estimation_id}/resume-stream",
        json={"decision": decision},
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def graph_progress(
    estimation_id: str,
    *,
    api_root: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    root = (api_root or get_api_root_url()).rstrip("/")
    response = httpx.get(
        f"{root}{GRAPH_PREFIX}/{estimation_id}/progress",
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def graph_state(
    estimation_id: str,
    *,
    api_root: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    root = (api_root or get_api_root_url()).rstrip("/")
    response = httpx.get(
        f"{root}{GRAPH_PREFIX}/{estimation_id}/state",
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def graph_proposal(
    estimation_id: str,
    *,
    api_root: str | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    root = (api_root or get_api_root_url()).rstrip("/")
    response = httpx.post(
        f"{root}{GRAPH_PREFIX}/{estimation_id}/proposal",
        headers=_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def activity_by_node(activity: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for entry in activity:
        grouped.setdefault(entry.get("node", ""), []).append(entry.get("message", ""))
    return grouped


def estimate_rows(estimate: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for module in (estimate or {}).get("modules") or []:
        for task in module.get("tasks") or []:
            rows.append(
                {
                    "module": module.get("name"),
                    "task": task.get("name"),
                    "description": task.get("description"),
                    "estimated_hours": task.get("estimated_hours"),
                    "reliability": task.get("reliability"),
                    "has_match": task.get("has_match"),
                }
            )
    return rows


def rows_to_estimate_overrides(rows: list[dict[str, Any]]) -> dict[str, Any]:
    modules: dict[str, dict[str, Any]] = {}
    for row in rows:
        module_name = row.get("module") or "Module"
        bucket = modules.setdefault(module_name, {"name": module_name, "tasks": []})
        bucket["tasks"].append(
            {
                "name": row.get("task"),
                "description": row.get("description"),
                "estimated_hours": row.get("estimated_hours"),
                "reliability": row.get("reliability"),
                "has_match": row.get("has_match"),
            }
        )
    return {"modules": list(modules.values())}


def proposal_pdf_bytes(title: str, body_markdown: str) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, title or "Proposal")
    pdf.ln(4)
    pdf.set_font("Helvetica", size=11)
    for line in (body_markdown or "").splitlines():
        pdf.multi_cell(0, 6, line)
    buffer = io.BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()
