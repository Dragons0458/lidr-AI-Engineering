"""Persistent history and restoration for Session 12 RAG runs."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent.parent / "path_setup.py"))

import streamlit as st

from streamlit_ui.agent_estimation import restore_session_state, run_to_session_payload
from streamlit_ui.store import get_rag_estimation_run, list_rag_estimation_runs

st.set_page_config(page_title="Histórico RAG", page_icon="🕘", layout="wide")
st.title("Histórico RAG")
st.caption("Runs persistidos del wizard híbrido de Sesión 12.")

c1, c2 = st.columns(2)
status_label = c1.selectbox(
    "Estado",
    ["Todos", "draft", "structure_review", "hours_review", "failed", "confirmed"],
)
mode_label = c2.selectbox("Modo", ["Todos", "agentic", "deterministic"])
runs = list_rag_estimation_runs(
    status=None if status_label == "Todos" else status_label,
    mode=None if mode_label == "Todos" else mode_label,
)

if not runs:
    st.info("No hay runs para los filtros seleccionados.")
    st.stop()

summary = [
    {
        "id": run["id"],
        "fecha": run["created_at"],
        "estado": run["status"],
        "modo": run["mode"],
        "perfil estructura": (
            (run.get("structure_profile_snapshot") or {}).get("name")
            or "Configuración del servicio"
        ),
        "perfil horas": (
            (run.get("hours_profile_snapshot") or {}).get("name")
            or "Configuración del servicio"
        ),
        "horas": run.get("total_hours"),
        "coste EUR": run.get("total_cost_eur"),
    }
    for run in runs
]
st.dataframe(summary, use_container_width=True, hide_index=True)

selected_id = st.selectbox(
    "Ver detalle",
    [run["id"] for run in runs],
    format_func=lambda run_id: next(
        f"#{run_id} · {run['created_at']} · {run['status']}"
        for run in runs
        if run["id"] == run_id
    ),
)
run = get_rag_estimation_run(selected_id)
if run is None:
    st.error("El run ya no existe.")
    st.stop()

if run["status"] != "confirmed":
    if st.button("Restaurar y continuar", type="primary"):
        restore_session_state(st.session_state, run_to_session_payload(run))
        st.switch_page("pages/5_RAG_Estimacion.py")
else:
    st.info("Run confirmado: vista de solo lectura.")

st.subheader(f"Run #{run['id']}")
meta1, meta2, meta3 = st.columns(3)
meta1.metric("Estado", run["status"])
meta2.metric("Modo", run["mode"])
meta3.metric("Coste", f"{run.get('total_cost_eur') or 0:.2f} €")

sections = [
    ("Transcript", run.get("transcript")),
    ("Reformulación", run.get("reformulation_payload")),
    ("Estructura propuesta", run.get("structure_response")),
    ("Estructura revisada", run.get("reviewed_structure")),
    ("Task-hours y traza de recovery", run.get("task_hours_response")),
    ("Hallucination gate", run.get("gate_report")),
    ("Breakdown final", run.get("final_rows")),
    ("One-shot", run.get("one_shot_result")),
    ("Snapshot perfil de estructura", run.get("structure_profile_snapshot")),
    ("Snapshot perfil de horas", run.get("hours_profile_snapshot")),
]
for title, value in sections:
    with st.expander(title, expanded=title in {"Transcript", "Breakdown final"}):
        if isinstance(value, (dict, list)):
            st.json(value)
        elif value:
            st.write(value)
        else:
            st.caption("Sin datos.")

if run.get("last_error"):
    st.error(f"Último error: {run['last_error']}")
