"""Session 12 handwritten-agent profile console."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent.parent / "path_setup.py"))

import streamlit as st

from streamlit_ui.agents import AgentProfile, avatar_data_uri, profile_summary
from streamlit_ui.graph_flow import GRAPH_NODES
from streamlit_ui.common import (
    INHERITED_MODEL_LABEL,
    fetch_available_agent_models,
    get_api_base_url,
)
from streamlit_ui.store import (
    create_agent_profile,
    delete_agent_profile,
    list_agent_profiles,
    set_default_agent_profile,
    update_agent_profile,
)

st.set_page_config(page_title="Agentes", page_icon="🤖", layout="wide")
st.title("Agentes")
st.caption(
    "Perfiles locales para el agente manual de estructura y recuperación de horas."
)

profiles = list_agent_profiles()
api_base = get_api_base_url()
catalog = fetch_available_agent_models(api_base)

if profiles:
    st.subheader("Perfiles handwritten")
    for profile in profiles:
        left, right = st.columns([1, 5])
        with left:
            avatar = avatar_data_uri(profile)
            if avatar:
                st.image(avatar, width=96)
            else:
                st.markdown("## 🤖")
        with right:
            default_label = " · **default**" if profile["is_default"] else ""
            st.markdown(f"### {profile['name']}{default_label}")
            st.caption(profile_summary(profile))
            if profile["persona"]:
                st.write(profile["persona"])
else:
    st.info("No hay perfiles. El wizard usará la configuración del servicio.")

st.divider()
st.subheader("Crear o editar")
profile_by_id = {profile["id"]: profile for profile in profiles}
options = [None, *profile_by_id]
selected_id = st.selectbox(
    "Perfil",
    options,
    format_func=lambda value: (
        "Nuevo perfil" if value is None else profile_by_id[value]["name"]
    ),
)
selected = profile_by_id.get(selected_id)
current_config = (selected or {}).get("config_payload") or {}
saved_model = str(current_config.get("model") or "")
models = [*catalog]
if saved_model and saved_model not in models:
    models.insert(0, saved_model)
model_options = ["", *models]

with st.form(f"agent_profile_{selected_id or 'new'}"):
    name = st.text_input(
        "Nombre", value=(selected or {}).get("name", ""), max_chars=120
    )
    model = st.selectbox(
        "Modelo",
        model_options,
        index=model_options.index(saved_model) if saved_model in model_options else 0,
        format_func=lambda value: (
            INHERITED_MODEL_LABEL
            if not value
            else f"{value} (no disponible)"
            if value not in catalog
            else value
        ),
    )
    effort_options = ["", "minimal", "low", "medium", "high"]
    effort = st.selectbox(
        "Reasoning effort",
        effort_options,
        index=effort_options.index(str(current_config.get("reasoning_effort") or "")),
        format_func=lambda value: INHERITED_MODEL_LABEL if not value else value,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        max_iterations = st.number_input(
            "Máximo de iteraciones (0 = heredar)",
            min_value=0,
            max_value=20,
            value=int(current_config.get("max_iterations") or 0),
            help="Default estático aprobado: 10.",
        )
    with c2:
        search_top_k = st.number_input(
            "Search top-k (0 = heredar)",
            min_value=0,
            max_value=30,
            value=int(current_config.get("search_top_k") or 0),
            help="Default estático aprobado: 5.",
        )
    with c3:
        distance_override = "search_distance_threshold" in current_config
        use_distance = st.checkbox(
            "Sobrescribir distancia",
            value=distance_override,
            help="Default estático aprobado: 0.45.",
        )
        search_distance = st.number_input(
            "Distancia",
            min_value=0.0,
            max_value=2.0,
            value=float(current_config.get("search_distance_threshold", 0.45)),
            step=0.05,
            disabled=not use_distance,
        )
    persona = st.text_area(
        "Persona",
        value=(selected or {}).get("persona", ""),
        max_chars=2000,
        height=150,
    )
    avatar = st.file_uploader(
        "Avatar (PNG, JPEG, GIF o WEBP)",
        type=["png", "jpg", "jpeg", "gif", "webp"],
    )
    remove_avatar = st.checkbox(
        "Eliminar avatar actual",
        value=False,
        disabled=not bool((selected or {}).get("avatar_bytes")),
    )
    is_default = st.checkbox(
        "Perfil por defecto", value=bool((selected or {}).get("is_default"))
    )
    submitted = st.form_submit_button("Guardar", type="primary")

if submitted:
    config = {
        "model": model or None,
        "reasoning_effort": effort or None,
        "max_iterations": int(max_iterations) or None,
        "search_top_k": int(search_top_k) or None,
        "search_distance_threshold": float(search_distance) if use_distance else None,
    }
    avatar_bytes = (selected or {}).get("avatar_bytes")
    avatar_filename = (selected or {}).get("avatar_filename")
    avatar_content_type = (selected or {}).get("avatar_content_type")
    if remove_avatar:
        avatar_bytes = avatar_filename = avatar_content_type = None
    elif avatar is not None:
        avatar_bytes = avatar.getvalue()
        avatar_filename = avatar.name
        avatar_content_type = avatar.type
    try:
        candidate = AgentProfile(
            id=selected_id,
            name=name,
            persona=persona,
            config=config,
            is_default=is_default,
            avatar_filename=avatar_filename,
            avatar_content_type=avatar_content_type,
            avatar_bytes=avatar_bytes,
        )
        if selected_id is None:
            create_agent_profile(candidate)
        else:
            update_agent_profile(selected_id, candidate)
        st.success("Perfil guardado.")
        st.rerun()
    except (ValueError, TypeError) as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo guardar el perfil: {exc}")

if selected_id is not None:
    d1, d2 = st.columns(2)
    with d1:
        if st.button("Marcar como default", disabled=bool(selected["is_default"])):
            set_default_agent_profile(selected_id)
            st.rerun()
    with d2:
        confirm_delete = st.checkbox(
            "Confirmo que quiero eliminarlo", key=f"delete_confirm_{selected_id}"
        )
        if st.button("Eliminar perfil", disabled=not confirm_delete):
            delete_agent_profile(selected_id)
            st.rerun()

st.divider()
st.subheader("Agentes del grafo (S13)")
st.caption("Catálogo didáctico del flujo multiagente con gates humanos.")
for node in GRAPH_NODES:
    character = node.get("character") or {}
    st.markdown(
        f"**{character.get('avatar', '🤖')} {node['label']}** — {node['role']} "
        f"(`{node.get('config_key') or '—'}`)"
    )
    st.caption(node["explanation"])
st.page_link("pages/9_Grafo_Agentes.py", label="Abrir wizard del grafo", icon="🕸")

st.divider()
st.subheader("Actor-Critic-Boss (solo lectura)")
st.write(
    "Este perfil configura el agente manual de Sesión 12. La orquestación "
    "Actor-Critic-Boss se consulta y configura en las vistas existentes."
)
c1, c2 = st.columns(2)
with c1:
    st.link_button("💬 Conversación", "/Conversacion", use_container_width=True)
with c2:
    st.link_button("⚙️ Ajustes IA", "/Ajustes_IA", use_container_width=True)
