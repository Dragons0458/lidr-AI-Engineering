"""Shared helpers for Streamlit frontends (errors, env display, API responses)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

MIN_DESCRIPTION_CHARS = 10
DEFAULT_API_BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_AGENT_MODELS = ("gpt-5", "gpt-5-mini", "gpt-4o", "gpt-4o-mini")
INHERITED_MODEL_LABEL = "Por defecto del servicio"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

_GUARDRAIL_LABELS: dict[str, str] = {
    "moderation": "Moderación de contenido",
    "prompt_injection": "Manipulación de instrucciones",
    "pii": "Datos personales (PII)",
}


def env_display(key: str, default: str = "—") -> str:
    value = os.getenv(key, "").strip()
    return value if value else default


def get_api_base_url() -> str:
    env_url = os.getenv("ESTIMATION_API_BASE_URL", DEFAULT_API_BASE_URL)
    try:
        return str(st.secrets.get("ESTIMATION_API_BASE_URL", env_url))
    except StreamlitSecretNotFoundError:
        return env_url


def get_api_root_url(base_url: str | None = None) -> str:
    """Strip ``/api/v1`` suffix for endpoints mounted at the API root (e.g. /embeddings)."""
    base = (base_url or get_api_base_url()).rstrip("/")
    if base.endswith("/api/v1"):
        return base.rsplit("/api/v1", 1)[0]
    return base


def get_estimate_api_key() -> str:
    """Return the Session 9 estimate API key from env or Streamlit secrets."""
    env_key = os.getenv("ESTIMATE_API_KEY", "").strip()
    try:
        return str(st.secrets.get("ESTIMATE_API_KEY", env_key))
    except StreamlitSecretNotFoundError:
        return env_key


def get_retrieval_api_key() -> str:
    """Return the Session 9 retrieval API key from env or Streamlit secrets."""
    env_key = os.getenv("RETRIEVAL_API_KEY", "").strip()
    try:
        return str(st.secrets.get("RETRIEVAL_API_KEY", env_key))
    except StreamlitSecretNotFoundError:
        return env_key


@st.cache_data(ttl=15)
def fetch_effective_primary_model(api_base_url: str) -> str | None:
    try:
        response = httpx.get(
            f"{api_base_url.rstrip('/')}/config/models",
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()["models"]["PRIMARY_MODEL"]["effective"]
    except (httpx.HTTPError, KeyError, TypeError):
        return None


def _is_openai_model(model: str) -> bool:
    try:
        from app.foundation.llm.wrapper import provider_from_model

        return provider_from_model(model) == "openai"
    except ImportError:
        name = model.split("/", 1)[-1].lower()
        return name.startswith(("gpt", "o1", "o3"))


@st.cache_data(ttl=30)
def fetch_available_agent_models(
    api_base_url: str,
    saved_model: str | None = None,
    *,
    timeout: float = 5.0,
) -> list[str]:
    """Return the OpenAI Responses-compatible catalog with a stable fallback."""
    try:
        response = httpx.get(
            f"{api_base_url.rstrip('/')}/config/models",
            timeout=timeout,
        )
        response.raise_for_status()
        raw_models = response.json().get("available_models") or []
        models = [
            str(model)
            for model in raw_models
            if isinstance(model, str) and _is_openai_model(model)
        ]
        if not models:
            models = list(DEFAULT_AGENT_MODELS)
    except (httpx.HTTPError, AttributeError, KeyError, TypeError, ValueError):
        models = list(DEFAULT_AGENT_MODELS)
    saved = (saved_model or "").strip()
    if saved and saved not in models:
        models.insert(0, saved)
    return list(dict.fromkeys(models))


def agent_model_label(model: str, available_models: list[str]) -> str:
    if not model:
        return INHERITED_MODEL_LABEL
    return model if model in available_models else f"{model} (no disponible)"


def parse_error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        text = response.text.strip()
        return text if text else "Sin cuerpo de respuesta."

    detail = body.get("detail")
    guardrail = format_guardrail_detail(detail)
    if guardrail:
        return guardrail

    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        lines = []
        for item in detail:
            if not isinstance(item, dict):
                continue
            loc = " → ".join(str(part) for part in item.get("loc", ()))
            msg = item.get("msg", "")
            lines.append(f"- **{loc}**: {msg}" if loc else f"- {msg}")
        return "\n".join(lines) if lines else json.dumps(body, ensure_ascii=False)
    if isinstance(detail, dict):
        return json.dumps(detail, ensure_ascii=False, indent=2)
    return json.dumps(body, ensure_ascii=False, indent=2)


def format_guardrail_detail(detail: Any) -> str | None:
    if not isinstance(detail, dict):
        return None
    reason = detail.get("reason")
    message = detail.get("message")
    if not reason or not message:
        return None
    label = _GUARDRAIL_LABELS.get(str(reason), str(reason))
    tips = {
        "moderation": "Reformula el texto sin contenido sensible o ofensivo.",
        "prompt_injection": "Evita frases del tipo «ignore previous instructions» o etiquetas `<system>`.",
        "pii": "Quita emails, IBAN o teléfonos de la descripción y de los adjuntos.",
    }
    tip = tips.get(str(reason), "Revisa la descripción y los archivos adjuntos.")
    return f"**{label}**\n\n{message}\n\n_{tip}_"


def format_api_error(exc: httpx.HTTPError, *, api_base_url: str) -> str:
    """Turn httpx/FastAPI errors into a short message for the chat UI."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        detail = parse_error_detail(exc.response)
        if status == 400:
            return f"**Solicitud rechazada (400)**\n\n{detail}"
        if status == 401:
            return (
                "**No autorizado (401)** — clave API inválida o ausente.\n\n"
                f"{detail}\n\n"
                "_Configura `ESTIMATE_API_KEY` / `RETRIEVAL_API_KEY` en `.env` o secrets._"
            )
        if status == 422:
            return (
                "**Solicitud rechazada (422)** — la API no pudo validar el mensaje.\n\n"
                f"{detail}\n\n"
                f"_Consejo: la descripción debe tener al menos {MIN_DESCRIPTION_CHARS} "
                "caracteres (p. ej. un resumen de reunión, no solo un saludo)._"
            )
        if status == 415:
            return f"**Tipo de archivo no admitido (415)**\n\n{detail}"
        if status == 429:
            retry = exc.response.headers.get("Retry-After", "60")
            return (
                f"**Límite de peticiones (429)** — espera {retry}s antes de reintentar.\n\n"
                f"{detail}"
            )
        if status == 502:
            return f"**Error del pipeline LLM (502)**\n\n{detail}"
        if status == 500:
            return f"**Error del servidor (500)**\n\n{detail}"
        return f"**Error HTTP {status}**\n\n{detail}"

    return (
        f"No se pudo conectar con la API en `{api_base_url}`.\n\n"
        "Comprueba que `uvicorn` esté en marcha y que la URL del sidebar sea correcta.\n\n"
        f"_Detalle técnico: {exc}_"
    )


def render_structured_phases(result: dict[str, Any]) -> None:
    """Render EstimationResult phases as a Streamlit table."""
    import streamlit as st

    phases = result.get("phases") or []
    if not phases:
        return
    st.markdown(f"**{result.get('summary', '')}**")
    st.caption(
        f"Confianza: {result.get('confidence_pct')}% · "
        f"Total: {result.get('total_hours')} h · "
        f"Coste: {result.get('total_cost_eur')} EUR"
    )
    st.dataframe(
        [
            {
                "Fase": phase.get("name"),
                "Base (h)": phase.get("base_hours"),
                "Buffer (h)": phase.get("buffer_hours"),
                "Equipo": phase.get("team"),
                "Resumen": phase.get("summary"),
            }
            for phase in phases
        ],
        use_container_width=True,
        hide_index=True,
    )


def resolve_sidebar_model(*, response_model: str | None = None) -> str:
    """Model shown in sidebar: last API response, else PRIMARY_MODEL from env."""
    if response_model and response_model.strip() and response_model != "-":
        return response_model
    return env_display("PRIMARY_MODEL", "—")
