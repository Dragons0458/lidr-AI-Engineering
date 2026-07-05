"""RAG Estimation wizard — Session 10 two-phase flow (structure → hours → review)."""

from __future__ import annotations

import runpy
import uuid
from pathlib import Path
from typing import Any

runpy.run_path(str(Path(__file__).resolve().parent.parent / "path_setup.py"))

import httpx
import pandas as pd
import streamlit as st

from streamlit_ui.common import (
    format_api_error,
    get_api_root_url,
    get_estimate_api_key,
)
from streamlit_ui.rag import verify_estimate

st.set_page_config(page_title="RAG Estimación", page_icon="📋", layout="wide")

api_root = get_api_root_url()
estimate_key = get_estimate_api_key()
headers = {"X-API-Key": estimate_key} if estimate_key else {}

_DEFAULTS = {
    "rag_transcript": "",
    "rag_reformulation": None,
    "rag_structure": None,
    "rag_structure_rows": None,
    "rag_hours": None,
    "rag_gate_report": None,
    "rag_final_rows": None,
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _clear_downstream(*keys: str) -> None:
    for key in keys:
        st.session_state[key] = _DEFAULTS[key]


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    if not estimate_key:
        st.error("Configura `ESTIMATE_API_KEY` en `.env` o `.streamlit/secrets.toml`.")
        return None
    try:
        response = httpx.post(
            f"{api_root.rstrip('/')}{path}",
            json=payload,
            headers=headers,
            timeout=600.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        st.error(format_api_error(exc, api_base_url=api_root))
        return None


def _structure_rows_from_estimate(estimate: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for module in estimate.get("modules", []):
        for task in module.get("tasks", []):
            rows.append(
                {
                    "module": module.get("name", ""),
                    "module_description": module.get("description") or "",
                    "task": task.get("name", ""),
                    "description": task.get("description") or "",
                }
            )
    return rows


def _modules_payload_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    modules: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("module", "")).strip()
        if not name:
            continue
        bucket = modules.setdefault(name, {"name": name, "tasks": []})
        task_name = str(row.get("task", "")).strip()
        if task_name:
            bucket["tasks"].append(
                {
                    "name": task_name,
                    "description": str(row.get("description") or "").strip() or None,
                }
            )
    return list(modules.values())


st.title("RAG Estimación")
st.caption(
    "Wizard S10: transcript → reformulación → estructura (revisión humana) → "
    "horas por tarea → revisión final."
)

if not estimate_key:
    st.warning(
        "Falta `ESTIMATE_API_KEY`. Los endpoints `/v1/estimate/*` requieren autenticación."
    )

with st.expander("Modo one-shot (comparación)"):
    if st.button("Ejecutar pipeline completo", key="oneshot"):
        transcript = st.session_state.rag_transcript or ""
        if len(transcript) < 100:
            st.error("El transcript debe tener al menos 100 caracteres.")
        else:
            result = _post(
                "/v1/estimate/from-transcript",
                {"transcript": transcript, "idempotency_key": str(uuid.uuid4())},
            )
            if result:
                st.json(result)

st.divider()
st.subheader("1. Transcript")
transcript = st.text_area(
    "Transcripción de la reunión",
    value=st.session_state.rag_transcript,
    height=200,
    key="transcript_input",
)
if st.button("Empezar", type="primary"):
    if len(transcript) < 100:
        st.error("Mínimo 100 caracteres.")
    else:
        st.session_state.rag_transcript = transcript
        _clear_downstream(
            "rag_reformulation",
            "rag_structure",
            "rag_structure_rows",
            "rag_hours",
            "rag_gate_report",
            "rag_final_rows",
        )
        result = _post("/v1/estimate/stages/reformulate", {"transcript": transcript})
        if result:
            st.session_state.rag_reformulation = result
            st.success("Reformulación completada.")

if st.session_state.rag_reformulation:
    st.divider()
    st.subheader("2. Reformulación")
    ref = st.session_state.rag_reformulation
    query = ref.get("query", {})
    col1, col2 = st.columns(2)
    with col1:
        st.json(query)
    with col2:
        st.markdown("**search_text** (para embedding)")
        st.code(ref.get("search_text", ""))

    st.divider()
    st.subheader("3. Estructura (sin horas)")
    if st.button("Generar estructura"):
        _clear_downstream(
            "rag_structure_rows", "rag_hours", "rag_gate_report", "rag_final_rows"
        )
        result = _post("/v1/estimate/stages/structure", {"query": query})
        if result:
            st.session_state.rag_structure = result
            st.session_state.rag_structure_rows = _structure_rows_from_estimate(
                result.get("estimate", {})
            )
            st.rerun()

if st.session_state.rag_structure_rows is not None:
    st.caption("Revisa y edita módulos/tareas antes de estimar horas.")
    edited_structure = st.data_editor(
        st.session_state.rag_structure_rows,
        num_rows="dynamic",
        use_container_width=True,
        key="structure_editor",
    )
    st.session_state.rag_structure_rows = edited_structure

    st.divider()
    st.subheader("4. Horas por tarea")
    if st.button("Estimar horas"):
        _clear_downstream("rag_hours", "rag_gate_report", "rag_final_rows")
        modules = _modules_payload_from_rows(edited_structure)
        if not modules:
            st.error("Añade al menos un módulo con tareas.")
        else:
            result = _post("/v1/estimate/tasks/hours", {"modules": modules})
            if result:
                st.session_state.rag_hours = result
                st.rerun()

if st.session_state.rag_hours:
    hours = st.session_state.rag_hours.get("tasks", [])
    rows: list[dict[str, Any]] = []
    for item in hours:
        rows.append(
            {
                "module": item.get("module"),
                "task": item.get("task"),
                "estimated_hours": item.get("estimated_hours"),
                "hours_range": (
                    f"{item['hours_range']['low']}–{item['hours_range']['high']} h"
                    if item.get("hours_range")
                    else ""
                ),
                "contradiction_reason": (
                    item["hours_range"].get("reason", "")
                    if item.get("hours_range")
                    else ""
                ),
                "reliability": item.get("reliability"),
                "dispersion": item.get("dispersion"),
                "has_match": item.get("has_match", True),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.apply(
            lambda row: [
                "background-color: #ffcccc" if not row.get("has_match") else ""
                for _ in row
            ],
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )
    for item in hours:
        if not item.get("has_match"):
            st.warning(
                f"Sin match histórico: **{item.get('module')} → {item.get('task')}**"
            )
        if item.get("hours_range"):
            hr = item["hours_range"]
            st.info(
                f"Contradicción · **{item.get('task')}**: "
                f"{hr.get('low')}–{hr.get('high')} h — {hr.get('reason', '')}"
            )
        neighbors = item.get("neighbors") or []
        if neighbors:
            with st.expander(f"Vecinos · {item.get('task')}"):
                st.json(neighbors)

    if st.session_state.rag_final_rows is None:
        st.session_state.rag_final_rows = [
            {
                "module": r["module"],
                "task": r["task"],
                "estimated_hours": r.get("estimated_hours") or 0,
                "hourly_rate_eur": 80,
            }
            for r in rows
        ]

    st.divider()
    st.subheader("5. Hallucination gate (opcional)")
    if st.button("Verificar líneas (sin juez LLM)"):
        modules = _modules_payload_from_rows(st.session_state.rag_structure_rows or [])
        estimate_stub = {
            "confidence": "high",
            "reasoning": "wizard review",
            "modules": [
                {
                    "name": m["name"],
                    "tasks": [
                        {
                            "name": t["name"],
                            "description": t.get("description"),
                            "engineer_days": None,
                            "grounded": False,
                            "sources": [],
                        }
                        for t in m.get("tasks", [])
                    ],
                }
                for m in modules
            ],
        }
        try:
            st.session_state.rag_gate_report = verify_estimate(
                api_root,
                estimate=estimate_stub,
                kept_chunks=[],
                estimate_key=estimate_key,
                use_judge=False,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(format_api_error(exc, api_base_url=api_root))

    if st.session_state.rag_gate_report:
        report = st.session_state.rag_gate_report
        st.metric("Grounded", report.get("grounded_lines", 0))
        st.metric("Degraded", report.get("degraded_lines", 0))
        st.metric("Insufficient", report.get("insufficient_lines", 0))
        for line in report.get("lines", []):
            st.caption(
                f"**{line.get('module')} → {line.get('component')}** · "
                f"`{line.get('status')}` · {line.get('reason', '')}"
            )

    st.divider()
    st.subheader("6. Revisión final")
    final_edited = st.data_editor(
        st.session_state.rag_final_rows,
        num_rows="dynamic",
        use_container_width=True,
        key="final_hours_editor",
    )
    st.session_state.rag_final_rows = final_edited
    total_hours = sum(int(r.get("estimated_hours") or 0) for r in final_edited)
    total_days = round(total_hours / 8, 1) if total_hours else 0
    st.metric("Total horas", total_hours)
    st.metric("Total engineer-days (8h)", total_days)
    if st.button("Confirmar estimación"):
        st.success(
            f"Estimación confirmada: {total_days} engineer-days ({total_hours} h)."
        )
