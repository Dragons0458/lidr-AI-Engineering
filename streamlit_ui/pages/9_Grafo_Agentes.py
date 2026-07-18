"""Session 13 multi-agent graph wizard (human gates + live activity panel)."""

from __future__ import annotations

import runpy
import uuid
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent.parent / "path_setup.py"))

import pandas as pd
import streamlit as st

from streamlit_ui.common import get_api_root_url, get_estimate_api_key
from streamlit_ui.graph_flow import (
    GRAPH_NODES,
    activity_by_node,
    estimate_rows,
    graph_progress,
    graph_proposal,
    graph_resume_stream,
    graph_start_stream,
    graph_state,
    proposal_pdf_bytes,
    rows_to_estimate_overrides,
)
from streamlit_ui.store import (
    create_graph_estimation_run,
    list_graph_estimation_runs,
    update_graph_estimation_run,
)

st.set_page_config(page_title="Grafo Agentes", page_icon="🕸", layout="wide")
st.title("Grafo multiagente (Sesión 13)")
st.caption("Wizard por gates humanos con panel de actividad en vivo.")

api_root = get_api_root_url()
api_key = get_estimate_api_key()

if "graph_run_id" not in st.session_state:
    st.session_state.graph_run_id = None
if "graph_estimation_id" not in st.session_state:
    st.session_state.graph_estimation_id = None


def _sync_local_run(state: dict) -> None:
    run_id = st.session_state.graph_run_id
    if run_id is None:
        return
    update_graph_estimation_run(
        run_id,
        graph_state=state,
        complexity=state.get("complexity"),
        structure=state.get("structure"),
        estimate=state.get("estimate"),
        analysis_report=state.get("analysis_report"),
        proposal=state.get("proposal"),
        status=state.get("status"),
    )


with st.sidebar:
    st.subheader("Histórico local")
    runs = list_graph_estimation_runs(limit=20)
    for run in runs:
        label = f"{run['estimation_id'][:12]}… · {run.get('status') or '—'}"
        if st.button(label, key=f"open_graph_run_{run['id']}"):
            st.session_state.graph_run_id = run["id"]
            st.session_state.graph_estimation_id = run["estimation_id"]
            st.rerun()

tab_start, tab_live, tab_gate1, tab_gate2, tab_done = st.tabs(
    [
        "1. Inicio",
        "2. Panel en vivo",
        "3. Gate estructura",
        "4. Gate final",
        "5. Completado",
    ]
)

with tab_start:
    transcript = st.text_area("Transcripción", height=220, placeholder="Pega el brief…")
    if st.button("Arrancar grafo", type="primary", disabled=not transcript.strip()):
        estimation_id = f"st-{uuid.uuid4()}"
        try:
            graph_start_stream(
                transcript.strip(),
                estimation_id,
                api_root=api_root,
                api_key=api_key,
            )
            run_id = create_graph_estimation_run(
                estimation_id=estimation_id,
                transcript=transcript.strip(),
                status="running",
            )
            st.session_state.graph_run_id = run_id
            st.session_state.graph_estimation_id = estimation_id
            st.success(f"Grafo arrancado: `{estimation_id}`")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo arrancar el grafo: {exc}")

with tab_live:
    estimation_id = st.session_state.graph_estimation_id
    if not estimation_id:
        st.info("Arranca un run desde la pestaña Inicio.")
    else:

        @st.fragment(run_every=2)
        def live_panel():
            try:
                progress = graph_progress(
                    estimation_id, api_root=api_root, api_key=api_key
                )
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return
            _sync_local_run(progress)
            grouped = activity_by_node(progress.get("activity") or [])
            cols = st.columns(len(GRAPH_NODES))
            for col, node in zip(cols, GRAPH_NODES, strict=False):
                with col:
                    character = node.get("character") or {}
                    st.markdown(f"### {character.get('avatar', '🤖')}")
                    st.caption(node["label"])
                    messages = grouped.get(node["key"], [])
                    if messages:
                        st.write(messages[-1])
            st.caption(f"Estado: **{progress.get('state')}**")
            if progress.get("state") != "running":
                st.success("Leg actual terminada — revisa el gate correspondiente.")

        live_panel()

with tab_gate1:
    estimation_id = st.session_state.graph_estimation_id
    if estimation_id:
        try:
            state = graph_state(estimation_id, api_root=api_root, api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            state = {}
        pending = state.get("pending_gate") or {}
        if pending.get("gate") == "structure_review":
            structure = pending.get("payload", {}).get("structure") or state.get(
                "structure"
            )
            modules = (structure or {}).get("modules") or []
            rows = []
            for module in modules:
                for task in module.get("tasks") or []:
                    rows.append(
                        {
                            "module": module.get("name"),
                            "task": task.get("name"),
                            "description": task.get("description"),
                        }
                    )
            edited = st.data_editor(
                pd.DataFrame(rows or [{"module": "", "task": "", "description": ""}])
            )
            if st.button("Aprobar estructura y continuar", type="primary"):
                modules_out: dict[str, dict] = {}
                for _, row in edited.iterrows():
                    module_name = str(row.get("module") or "Module")
                    bucket = modules_out.setdefault(
                        module_name, {"name": module_name, "tasks": []}
                    )
                    bucket["tasks"].append(
                        {
                            "name": row.get("task"),
                            "description": row.get("description"),
                        }
                    )
                decision = {"approved": True, "modules": list(modules_out.values())}
                try:
                    graph_resume_stream(
                        estimation_id,
                        decision,
                        api_root=api_root,
                        api_key=api_key,
                    )
                    st.success("Estructura enviada — sigue el panel en vivo.")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
        else:
            st.info("No hay gate de estructura pendiente.")

with tab_gate2:
    estimation_id = st.session_state.graph_estimation_id
    if estimation_id:
        try:
            state = graph_state(estimation_id, api_root=api_root, api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            state = {}
        pending = state.get("pending_gate") or {}
        if pending.get("gate") == "final_review":
            report = pending.get("payload", {}).get("analysis_report") or state.get(
                "analysis_report"
            )
            estimate = pending.get("payload", {}).get("estimate") or state.get(
                "estimate"
            )
            if report:
                st.metric("Confianza global", report.get("overall_confidence", "—"))
                st.metric(
                    "Tareas fundamentadas",
                    f"{round((report.get('grounded_task_ratio') or 0) * 100)}%",
                )
                for point in report.get("weak_points") or []:
                    st.warning(f"{point.get('area')}: {point.get('issue')}")
            rows = estimate_rows(estimate)
            edited = st.data_editor(
                pd.DataFrame(
                    rows or [{"module": "", "task": "", "estimated_hours": None}]
                ),
                column_config={
                    "estimated_hours": st.column_config.NumberColumn(
                        "Horas", min_value=0.0
                    )
                },
            )
            want_proposal = st.checkbox("Quiero propuesta comercial", value=True)
            if st.button("Validar estimación", type="primary"):
                decision = {
                    "validated": True,
                    "want_proposal": want_proposal,
                    "estimate_overrides": rows_to_estimate_overrides(
                        edited.to_dict(orient="records")
                    ),
                }
                try:
                    graph_resume_stream(
                        estimation_id,
                        decision,
                        api_root=api_root,
                        api_key=api_key,
                    )
                    st.success("Validación enviada — sigue el panel en vivo.")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
        else:
            st.info("No hay gate final pendiente.")

with tab_done:
    estimation_id = st.session_state.graph_estimation_id
    if estimation_id:
        try:
            state = graph_state(estimation_id, api_root=api_root, api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            state = {}
        if state.get("state") == "completed":
            estimate = state.get("estimate") or {}
            st.metric("Jornadas", estimate.get("total_engineer_days", "—"))
            st.metric("Confianza", estimate.get("confidence", "—"))
            proposal = state.get("proposal")
            if proposal:
                st.markdown(proposal)
                st.download_button(
                    "Descargar .md",
                    data=proposal,
                    file_name=f"{estimation_id}-proposal.md",
                )
                st.download_button(
                    "Descargar .pdf",
                    data=proposal_pdf_bytes("Propuesta", proposal),
                    file_name=f"{estimation_id}-proposal.pdf",
                    mime="application/pdf",
                )
            if st.button("Generar / regenerar propuesta"):
                try:
                    drafted = graph_proposal(
                        estimation_id, api_root=api_root, api_key=api_key
                    )
                    st.session_state[f"proposal_{estimation_id}"] = drafted
                    st.success("Propuesta generada.")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
        else:
            st.info("El grafo aún no ha completado.")
