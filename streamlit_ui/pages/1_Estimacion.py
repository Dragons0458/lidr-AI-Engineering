"""Transactional estimation page with local history (mirrors Rails estimations context)."""

from __future__ import annotations

import runpy
import time
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent.parent / "path_setup.py"))

import httpx
import streamlit as st

from app.domain.schemas.estimation import DetailLevel, OutputFormat, ProjectType
from streamlit_ui.common import (
    MIN_DESCRIPTION_CHARS,
    format_api_error,
    get_api_base_url,
    render_structured_phases,
)
from streamlit_ui.store import get_estimation, list_estimations, save_estimation

st.set_page_config(page_title="Estimación", page_icon="📝", layout="wide")

api_base_url = get_api_base_url()


def _estimation_cost_preview(payload: dict) -> str:
    cost = payload.get("cost_usd")
    if cost:
        return f"${float(cost):.4f}"
    return "—"


st.title("Estimación")
st.caption(f"Endpoint transaccional `POST {api_base_url.rstrip('/')}/estimate`")

with st.form("estimation_form"):
    description = st.text_area(
        "Descripción o resumen de reunión",
        height=180,
        placeholder="Describe el alcance del proyecto de software…",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        project_type = st.selectbox(
            "Tipo de proyecto",
            options=list(ProjectType),
            format_func=lambda value: value.value,
        )
        detail_level = st.selectbox(
            "Nivel de detalle",
            options=list(DetailLevel),
            format_func=lambda value: value.value,
        )
    with col_b:
        output_format = st.selectbox(
            "Formato de salida",
            options=list(OutputFormat),
            format_func=lambda value: value.value,
        )
        prompt_version = st.selectbox("Versión de prompt", options=["v1", "v2"])
    submitted = st.form_submit_button("Generar estimación", type="primary")

if submitted:
    if len(description.strip()) < MIN_DESCRIPTION_CHARS:
        st.error(
            f"La descripción debe tener al menos {MIN_DESCRIPTION_CHARS} caracteres."
        )
    else:
        endpoint = f"{api_base_url.rstrip('/')}/estimate"
        payload = {
            "description": description.strip(),
            "project_type": project_type.value,
            "detail_level": detail_level.value,
            "output_format": output_format.value,
        }
        with st.spinner("Generando estimación…"):
            start = time.perf_counter()
            try:
                response = httpx.post(
                    endpoint,
                    json=payload,
                    params={"prompt_version": prompt_version},
                    timeout=120.0,
                )
                response.raise_for_status()
                body = response.json()
                duration_ms = int((time.perf_counter() - start) * 1000)
                estimation_id = save_estimation(
                    description=description.strip(),
                    project_type=project_type.value,
                    detail_level=detail_level.value,
                    output_format=output_format.value,
                    response_payload=body,
                    prompt_version=prompt_version,
                    cached=bool(body.get("cache_hit")),
                )
                st.session_state.last_estimation_id = estimation_id
                st.session_state.last_estimation_body = body
                st.success(
                    f"Estimación guardada (#{estimation_id}) en {duration_ms} ms."
                )
            except httpx.HTTPError as exc:
                st.error(format_api_error(exc, api_base_url=api_base_url))

if st.session_state.get("last_estimation_body"):
    body = st.session_state.last_estimation_body
    st.subheader("Última respuesta")
    st.markdown(body.get("estimation", ""))
    cols = st.columns(4)
    cols[0].metric("Modelo", body.get("model", "—"))
    cols[1].metric("Caché", "Sí" if body.get("cache_hit") else "No")
    cols[2].metric("Coste USD", f"{body.get('cost_usd', 0):.4f}")
    cols[3].metric("Fuera de alcance", "Sí" if body.get("out_of_scope") else "No")

st.divider()
st.subheader("Histórico local")

history = list_estimations(limit=20)
if not history:
    st.caption("Sin estimaciones guardadas todavía.")
else:
    for row in history:
        payload = row.get("response_payload") or {}
        preview = (row.get("description") or "")[:80]
        cols = st.columns([4, 1, 1, 1])
        cols[0].write(
            f"**#{row['id']}** · {preview}… · `{row['project_type']}` · "
            f"cached={'sí' if row['cached'] else 'no'}"
        )
        cols[1].caption(_estimation_cost_preview(payload))
        cols[2].caption((row.get("created_at") or "")[:19])
        if cols[3].button("Ver", key=f"view_est_{row['id']}"):
            st.session_state.view_estimation_id = row["id"]
            st.rerun()

if st.session_state.get("view_estimation_id"):
    record = get_estimation(st.session_state.view_estimation_id)
    if record:
        payload = record.get("response_payload") or {}
        with st.expander(f"Estimación #{record['id']}", expanded=True):
            st.markdown(payload.get("estimation", ""))
            if payload.get("phases"):
                render_structured_phases(payload)
            st.json(payload)
