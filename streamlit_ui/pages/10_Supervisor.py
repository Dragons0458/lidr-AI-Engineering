"""Session 14 supervisor multi-agent wizard (routing + privilege + HITL)."""

from __future__ import annotations

import json
import runpy
import uuid
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent.parent / "path_setup.py"))

import pandas as pd
import streamlit as st

from streamlit_ui.common import get_api_root_url, get_estimate_api_key
from streamlit_ui.supervisor_flow import (
    load_sample_transcript,
    status_badge_label,
    supervisor_resume,
    supervisor_start,
    supervisor_state,
)

st.set_page_config(page_title="Supervisor", page_icon="🧭", layout="wide")
st.title("Supervisor multiagente (Sesión 14)")
st.caption(
    "Estrella con router LLM, privilegio exigible e intervención humana por señal."
)

api_root = get_api_root_url()
api_key = get_estimate_api_key()

if "supervisor_state" not in st.session_state:
    st.session_state.supervisor_state = None
if "supervisor_estimation_id" not in st.session_state:
    st.session_state.supervisor_estimation_id = None


def _badge(status: str | None) -> None:
    label = status_badge_label(status)
    colors = {
        "awaiting_human_review": "🟠",
        "validated": "🟢",
        "needs_review": "🟡",
        "rejected": "🔴",
    }
    st.markdown(f"### {colors.get(label, '⚪')} Estado: `{label}`")


tab_start, tab_review, tab_done, tab_recover = st.tabs(
    ["1. Inicio", "2. Revisión humana", "3. Completado", "4. Recuperar run"]
)

with tab_start:
    sample = st.selectbox(
        "Cargar transcripción de ejemplo",
        ["(ninguna)", "happy_path", "edge_case"],
    )
    default_text = ""
    if sample == "happy_path":
        default_text = load_sample_transcript("happy_path")
    elif sample == "edge_case":
        default_text = load_sample_transcript("edge_case")

    transcript = st.text_area("Transcripción", value=default_text, height=260)
    if st.button(
        "Arrancar supervisor", type="primary", disabled=len(transcript.strip()) < 100
    ):
        estimation_id = f"st-s14-{uuid.uuid4()}"
        try:
            state = supervisor_start(
                transcript.strip(),
                estimation_id=estimation_id,
                api_root=api_root,
                api_key=api_key,
            )
            st.session_state.supervisor_state = state
            st.session_state.supervisor_estimation_id = state["estimation_id"]
            st.success(f"Run: `{state['estimation_id']}`")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error al arrancar: {exc}")

with tab_review:
    state = st.session_state.supervisor_state
    if not state:
        st.info("Arranca un run o recupéralo desde la pestaña 4.")
    else:
        _badge(state.get("status"))
        pending = state.get("pending_review")
        if state.get("status") == "awaiting_human_review" and pending:
            st.write("**Razones:**")
            for reason in pending.get("reasons") or []:
                st.write(f"- `{reason}`")
            risk_flags = pending.get("risk_flags") or []
            if risk_flags:
                st.write("**Señales de riesgo:**")
                for flag in risk_flags:
                    st.write(f"- `{flag}`")
            st.write(
                f"Confianza: `{pending.get('confidence')}` "
                f"(umbral `{pending.get('threshold')}`)"
            )
            estimate = pending.get("estimate") or state.get("estimate") or {}
            st.json(estimate)

            action = st.radio(
                "Decisión", ["approve", "adjust", "reject"], horizontal=True
            )
            note = st.text_input("Nota (opcional)")
            overrides_raw = ""
            if action == "adjust":
                overrides_raw = st.text_area(
                    "estimate_overrides (JSON)",
                    value=json.dumps(
                        {"components": estimate.get("components") or []}, indent=2
                    ),
                    height=200,
                )
            if st.button("Enviar decisión", type="primary"):
                overrides = None
                if action == "adjust" and overrides_raw.strip():
                    overrides = json.loads(overrides_raw)
                try:
                    updated = supervisor_resume(
                        state["estimation_id"],
                        action,
                        estimate_overrides=overrides,
                        note=note or None,
                        api_root=api_root,
                        api_key=api_key,
                    )
                    st.session_state.supervisor_state = updated
                    st.success("Reanudado.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Error al reanudar: {exc}")
        elif state.get("status") == "needs_review":
            st.info(
                "`needs_review` es una advertencia del validador de coherencia "
                "(issues / fuera de banda), **no** la pausa HITL. "
                "El gate humano solo aparece con `awaiting_human_review`. "
                "Mira la pestaña **3. Completado** para el resultado."
            )
        else:
            st.write("No hay revisión humana pendiente.")

with tab_done:
    state = st.session_state.supervisor_state
    if not state:
        st.info("Sin run en sesión.")
    else:
        _badge(state.get("status"))
        if state.get("estimate"):
            st.subheader("Estimación")
            st.json(state["estimate"])
            st.write(f"Confianza: `{state.get('confidence')}`")

        st.subheader("Historial de enrutado")
        routing = state.get("routing_history") or []
        if routing:
            st.dataframe(pd.DataFrame(routing), use_container_width=True)
        else:
            st.caption("Sin filas de enrutado.")

        st.subheader("Auditoría de acciones")
        contributions = state.get("agent_contributions") or []
        if contributions:
            df = pd.DataFrame(contributions)
            if "outcome" in df.columns:
                denied = df["outcome"] == "denied"
                st.dataframe(
                    df.style.apply(
                        lambda col: (
                            [
                                "background-color: #fecaca" if denied.iloc[i] else ""
                                for i in range(len(col))
                            ]
                            if col.name == "outcome"
                            else [""] * len(col)
                        ),
                        axis=0,
                    ),
                    use_container_width=True,
                )
            else:
                st.dataframe(df, use_container_width=True)
            violations = state.get("privilege_violations") or []
            if violations:
                st.warning(
                    f"{len(violations)} denegación(es) de privilegio en el trail."
                )
        else:
            st.caption("Sin contribuciones.")

with tab_recover:
    estimation_id = st.text_input(
        "estimation_id",
        value=st.session_state.supervisor_estimation_id or "",
    )
    if st.button("Cargar estado", disabled=not estimation_id.strip()):
        try:
            state = supervisor_state(
                estimation_id.strip(),
                api_root=api_root,
                api_key=api_key,
            )
            st.session_state.supervisor_state = state
            st.session_state.supervisor_estimation_id = state["estimation_id"]
            st.success("Estado cargado.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo cargar: {exc}")
