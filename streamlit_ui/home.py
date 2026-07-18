"""Streamlit Home — entry point for the multipage estimator frontend."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "path_setup.py"))

import streamlit as st

from streamlit_ui.common import fetch_effective_primary_model, get_api_base_url

st.set_page_config(
    page_title="Estimador CAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

api_base_url = get_api_base_url()

with st.sidebar:
    st.header("Estado")
    primary = fetch_effective_primary_model(api_base_url)
    if primary:
        st.success(f"Modelo primario: `{primary}`")
    st.caption(f"API: `{api_base_url}`")

st.title("Estimador CAG")
st.markdown(
    "Frontend Streamlit con paridad funcional respecto a **estimator-web** (Rails): "
    "estimación transaccional, conversación multi-turno, RAG Chunking Lab y ajustes "
    "de modelos en runtime."
)

col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/1_Estimacion.py", label="Estimación", icon="📝")
    st.caption("POST `/api/v1/estimate` — formulario transaccional e histórico local.")

    st.page_link("pages/3_RAG_Lab.py", label="RAG Chunking Lab", icon="🧪")
    st.caption("Compara 8 estrategias de chunking sobre el corpus de presupuestos.")

    st.page_link("pages/5_RAG_Estimacion.py", label="RAG Estimación", icon="📋")
    st.caption("Wizard híbrido con agente, recovery y revisión humana (Sesión 12).")

    st.page_link("pages/7_Agentes.py", label="Agentes", icon="🤖")
    st.caption("Perfiles handwritten, personas, avatares y knobs del agente.")

    st.page_link("pages/9_Grafo_Agentes.py", label="Grafo Agentes", icon="🕸")
    st.caption("Wizard S13: grafo multiagente con gates humanos y panel en vivo.")

with col2:
    st.page_link("pages/2_Conversacion.py", label="Conversación", icon="💬")
    st.caption("Sesiones, adjuntos, memoria comprimida y Actor-Critic-Boss.")

    st.page_link("pages/4_Ajustes_IA.py", label="Ajustes IA", icon="⚙️")
    st.caption("Overrides runtime de modelos vía `GET/PUT /api/v1/config/models`.")

    st.page_link("pages/8_Historial_RAG.py", label="Histórico RAG", icon="🕘")
    st.caption("Runs incompletos restaurables y confirmaciones de solo lectura.")

st.divider()
st.markdown(
    "**Persistencia local:** SQLite en `STREAMLIT_DB_PATH` "
    "(por defecto `streamlit_ui/data/frontend.db`). Incluye estimaciones, sesiones, "
    "comparaciones, perfiles, avatares y runs RAG."
)
