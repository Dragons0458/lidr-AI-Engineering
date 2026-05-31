"""Streamlit chat UI that consumes the SSE estimation stream token by token."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType
from streamlit_common import MIN_DESCRIPTION_CHARS, env_display, format_api_error

DEFAULT_API_BASE_URL = "http://localhost:8000/api/v1"

load_dotenv(Path(__file__).resolve().parent / ".env")


def get_api_base_url() -> str:
    env_url = os.getenv("ESTIMATION_API_BASE_URL", DEFAULT_API_BASE_URL)
    try:
        return str(st.secrets.get("ESTIMATION_API_BASE_URL", env_url))
    except StreamlitSecretNotFoundError:
        return env_url


def stream_estimation(description: str, *, api_base_url: str) -> Iterator[str]:
    """POST to /estimate/stream and yield text chunks from SSE events."""
    endpoint = f"{api_base_url.rstrip('/')}/estimate/stream"
    payload = {
        "description": description,
        "project_type": ProjectType.WEB_SAAS.value,
        "detail_level": DetailLevel.MEDIUM.value,
        "output_format": OutputFormat.PHASES_TABLE.value,
        "evaluate": False,
    }
    with httpx.stream(
        "POST",
        endpoint,
        json=payload,
        timeout=httpx.Timeout(120.0, connect=10.0),
        headers={"Accept": "text/event-stream"},
    ) as response:
        if response.is_error:
            response.read()
            response.raise_for_status()
        current_event = "token"
        data_lines: list[str] = []
        for raw_line in response.iter_lines():
            if raw_line == "":
                if data_lines:
                    payload_text = "\n".join(data_lines)
                    data_lines = []
                    if current_event == "token":
                        yield payload_text
                    elif current_event == "error":
                        yield f"\n\n[error] {payload_text}"
                    elif current_event == "done":
                        return
                current_event = "token"
                continue
            if raw_line.startswith("event:"):
                current_event = raw_line[6:].strip()
            elif raw_line.startswith("data:"):
                data_lines.append(
                    raw_line[6:] if raw_line.startswith("data: ") else raw_line[5:]
                )


st.set_page_config(page_title="Estimator Stream", page_icon="📊")
st.title("Software Estimator (streaming)")
st.caption(
    "Respuestas en streaming vía SSE (`POST /api/v1/estimate/stream`). "
    f"Mínimo **{MIN_DESCRIPTION_CHARS} caracteres** por mensaje."
)

with st.sidebar:
    st.header("Configuración")
    api_base_url = st.text_input("API base URL", value=get_api_base_url())
    st.divider()
    st.subheader("Modelos (.env)")
    st.text_input("LLM_PROVIDER", value=env_display("LLM_PROVIDER"), disabled=True)
    st.text_input("PRIMARY_MODEL", value=env_display("PRIMARY_MODEL"), disabled=True)
    st.text_input(
        "FALLBACK_MODEL",
        value=env_display("FALLBACK_MODEL", "(sin fallback)"),
        disabled=True,
    )
    st.divider()
    st.subheader("Sesión 5 (.env)")
    st.caption(
        "El streaming usa `POST /estimate/stream` (markdown). "
        "Para ACB estructurado usa `streamlit_app.py` con sesiones."
    )
    st.divider()
    st.subheader("Caché (.env)")
    st.text_input("REDIS_URL", value=env_display("REDIS_URL"), disabled=True)
    st.text_input("CACHE_TTL", value=env_display("CACHE_TTL", "86400"), disabled=True)
    cache_on = os.getenv("CACHE_ENABLED", "true").strip().lower()
    st.text_input("CACHE_ENABLED", value=cache_on or "true", disabled=True)
    if not os.getenv("PRIMARY_MODEL", "").strip():
        st.warning(
            "PRIMARY_MODEL vacío: ejecuta Streamlit desde la raíz del proyecto "
            "o define variables en `.streamlit/secrets.toml`."
        )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(
    "Describe el proyecto o pega un resumen de reunión (mín. 10 caracteres)..."
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        if len(prompt.strip()) < MIN_DESCRIPTION_CHARS:
            full_response = (
                f"**Mensaje demasiado corto** ({len(prompt.strip())} caracteres).\n\n"
                f"La API exige al menos **{MIN_DESCRIPTION_CHARS}** en `description`. "
                "Escribe un resumen de reunión o alcance del proyecto."
            )
            placeholder.warning(full_response)
        else:
            try:
                for chunk in stream_estimation(prompt, api_base_url=api_base_url):
                    full_response += chunk
                    placeholder.markdown(full_response + "▍")
                placeholder.markdown(full_response)
            except httpx.HTTPError as exc:
                full_response = format_api_error(exc, api_base_url=api_base_url)
                placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
