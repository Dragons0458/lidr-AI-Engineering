"""Runtime model settings (mirrors Rails ai_settings/show)."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent.parent / "path_setup.py"))

import httpx
import streamlit as st

from streamlit_ui.common import fetch_effective_primary_model, get_api_base_url
from streamlit_ui.rag import (
    MODEL_KNOB_LABELS,
    RETRIEVAL_KNOB_LABELS,
    build_retrieval_update_payload,
    build_settings_update_payload,
)

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

st.divider()
st.subheader("Recuperación (runtime)")
retrieval_url = f"{api_base_url.rstrip('/')}/config/retrieval"


@st.cache_data(ttl=15)
def fetch_retrieval_config(api_base: str) -> dict | None:
    try:
        response = httpx.get(f"{api_base.rstrip('/')}/config/retrieval", timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


retrieval_config = fetch_retrieval_config(api_base_url)
if retrieval_config is None:
    st.warning("No se pudo cargar la configuración de recuperación.")
else:
    retrieval_snapshot: dict = retrieval_config.get("retrieval") or {}
    if "retrieval_selections" not in st.session_state:
        st.session_state.retrieval_selections = {}

    retrieval_touched: set[str] = set()
    search_meta = retrieval_snapshot.get("RETRIEVAL_SEARCH_MODE", {})
    search_effective = search_meta.get("effective", "vector")
    search_default = search_meta.get("default", "vector")
    st.markdown(
        f"**Search mode** · efectivo `{search_effective}` · default `{search_default}`"
    )
    search_mode = st.selectbox(
        "RETRIEVAL_SEARCH_MODE",
        options=["", "vector", "hybrid"],
        format_func=lambda v: f"Por defecto ({search_default})" if v == "" else v,
        index=["", "vector", "hybrid"].index(
            st.session_state.retrieval_selections.get("search_mode", "")
        ),
        key="retrieval_search_mode",
    )
    if search_mode != st.session_state.retrieval_selections.get("search_mode", ""):
        retrieval_touched.add("search_mode")
    st.session_state.retrieval_selections["search_mode"] = search_mode

    bool_fields = [
        ("rerank", "RERANKER_ENABLED"),
        ("routing_enabled", "RETRIEVAL_ROUTING_ENABLED"),
        ("query_transform_enabled", "QUERY_TRANSFORM_ENABLED"),
        ("temporal_decay_enabled", "TEMPORAL_DECAY_ENABLED"),
    ]
    bool_values: dict[str, bool | None] = {}
    for field, redis_key in bool_fields:
        meta = retrieval_snapshot.get(redis_key, {})
        knob = RETRIEVAL_KNOB_LABELS.get(
            redis_key, {"label": redis_key, "description": ""}
        )
        st.markdown(f"**{knob['label']}** `{redis_key}`")
        st.caption(knob["description"])
        effective = bool(meta.get("effective", False))
        default = bool(meta.get("default", False))
        choice = st.radio(
            f"Valor para {redis_key}",
            options=["default", "true", "false"],
            format_func=lambda v: {
                "default": f"Por defecto ({default})",
                "true": "Activado",
                "false": "Desactivado",
            }[v],
            index={"default": 0, "true": 1, "false": 2}.get(
                st.session_state.retrieval_selections.get(field, "default"), 0
            ),
            horizontal=True,
            key=f"retrieval_{field}",
            label_visibility="collapsed",
        )
        if choice != st.session_state.retrieval_selections.get(field, "default"):
            retrieval_touched.add(field)
        st.session_state.retrieval_selections[field] = choice
        bool_values[field] = None if choice == "default" else choice == "true"
        st.caption(f"Efectivo: `{effective}`")

    th_k_meta = retrieval_snapshot.get("TASK_HOURS_TOP_K", {})
    th_k = st.number_input(
        "TASK_HOURS_TOP_K (0 = default)",
        min_value=0,
        max_value=30,
        value=int(st.session_state.retrieval_selections.get("task_hours_top_k", 0)),
        key="retrieval_task_hours_top_k",
    )
    if th_k != st.session_state.retrieval_selections.get("task_hours_top_k", 0):
        retrieval_touched.add("task_hours_top_k")
    st.session_state.retrieval_selections["task_hours_top_k"] = th_k

    th_d_meta = retrieval_snapshot.get("TASK_HOURS_DISTANCE_THRESHOLD", {})
    th_d = st.number_input(
        "TASK_HOURS_DISTANCE_THRESHOLD (0 = default)",
        min_value=0.0,
        max_value=2.0,
        value=float(
            st.session_state.retrieval_selections.get(
                "task_hours_distance_threshold", 0.0
            )
        ),
        step=0.05,
        key="retrieval_task_hours_distance_threshold",
    )
    if th_d != st.session_state.retrieval_selections.get(
        "task_hours_distance_threshold", 0.0
    ):
        retrieval_touched.add("task_hours_distance_threshold")
    st.session_state.retrieval_selections["task_hours_distance_threshold"] = th_d
    st.caption(
        f"Efectivo top_k: `{th_k_meta.get('effective')}` · "
        f"threshold: `{th_d_meta.get('effective')}`"
    )

    if st.button("Guardar recuperación"):
        payload = build_retrieval_update_payload(
            search_mode=search_mode or None,
            rerank=bool_values["rerank"],
            routing_enabled=bool_values["routing_enabled"],
            query_transform_enabled=bool_values["query_transform_enabled"],
            temporal_decay_enabled=bool_values["temporal_decay_enabled"],
            task_hours_top_k=int(th_k) if th_k > 0 else None,
            task_hours_distance_threshold=float(th_d) if th_d > 0 else None,
            touched=retrieval_touched
            or set(bool_fields)
            | {"search_mode", "task_hours_top_k", "task_hours_distance_threshold"},
        )
        try:
            response = httpx.put(retrieval_url, json=payload, timeout=15.0)
            if response.status_code == 200:
                fetch_retrieval_config.clear()
                st.success("Configuración de recuperación guardada.")
                st.rerun()
            elif response.status_code in (400, 422):
                st.error("Cambio rechazado por la API.")
                st.json(response.json())
            elif response.status_code == 503:
                st.error("Redis no disponible; no se aplicaron overrides.")
            else:
                st.error(f"Error HTTP {response.status_code}")
        except httpx.HTTPError as exc:
            st.error(f"No se pudo guardar: {exc}")

st.divider()
if st.button("Guardar cambios", type="primary", key="save_models"):
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
