"""Runtime model settings (mirrors Rails ai_settings/show)."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent.parent / "path_setup.py"))

import httpx
import streamlit as st

from streamlit_ui.common import fetch_effective_primary_model, get_api_base_url
from streamlit_ui.rag import MODEL_KNOB_LABELS, build_settings_update_payload

st.set_page_config(page_title="Ajustes IA", page_icon="⚙️", layout="wide")

api_base_url = get_api_base_url()
config_url = f"{api_base_url.rstrip('/')}/config/models"


@st.cache_data(ttl=15)
def fetch_config(api_base: str) -> dict | None:
    try:
        response = httpx.get(f"{api_base.rstrip('/')}/config/models", timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


st.title("Ajustes IA")
st.caption(f"`GET/PUT {config_url}`")

config = fetch_config(api_base_url)
if config is None:
    st.error("No se pudo cargar la configuración de modelos desde la API.")
    st.stop()

available_models: list[str] = config.get("available_models") or []
models_snapshot: dict = config.get("models") or {}

if "settings_selections" not in st.session_state:
    st.session_state.settings_selections: dict[str, str] = {}

st.subheader("Modelos LLM (runtime)")
selections: dict[str, str] = {}

for key, meta in models_snapshot.items():
    knob = MODEL_KNOB_LABELS.get(key, {"label": key, "description": ""})
    effective = meta.get("effective", "")
    default = meta.get("default", "")
    overridden = meta.get("overridden", False)
    status = "override" if overridden else "default"
    st.markdown(f"**{knob['label']}** `{key}` · _{status}_")
    st.caption(knob["description"])
    options = [("", f"Por defecto ({default})")] + [
        (model, model) for model in available_models
    ]
    current = st.session_state.settings_selections.get(key, "")
    if not current and overridden:
        current = effective
    index = 0
    for idx, (value, _) in enumerate(options):
        if value == current or (not value and not overridden):
            index = idx
            break
    selected = st.selectbox(
        f"Valor para {key}",
        options=[label for _, label in options],
        index=index,
        key=f"knob_{key}",
        label_visibility="collapsed",
    )
    selected_value = options[[label for _, label in options].index(selected)][0]
    selections[key] = selected_value
    st.caption(f"Efectivo: `{effective}`")

st.divider()
st.subheader("Embeddings (solo lectura)")
st.text_input(
    "EMBEDDING_MODEL",
    value=config.get("embedding_model", ""),
    disabled=True,
)
st.caption(config.get("embedding_model_note", ""))

if st.button("Guardar cambios", type="primary"):
    payload = {"models": build_settings_update_payload(selections)}
    try:
        response = httpx.put(config_url, json=payload, timeout=15.0)
        if response.status_code == 200:
            st.session_state.settings_selections = selections
            fetch_config.clear()
            fetch_effective_primary_model.clear()
            st.success("Configuración guardada.")
            st.rerun()
        elif response.status_code in (400, 422):
            st.error("Cambio rechazado por la API.")
            st.json(response.json())
        elif response.status_code == 503:
            st.error("Servicio o Redis no disponible; no se aplicaron overrides.")
        else:
            st.error(f"Error HTTP {response.status_code}")
            st.text(response.text)
    except httpx.HTTPError as exc:
        st.error(f"No se pudo guardar: {exc}")

primary = fetch_effective_primary_model(api_base_url)
if primary:
    st.sidebar.success(f"Modelo primario efectivo: `{primary}`")
