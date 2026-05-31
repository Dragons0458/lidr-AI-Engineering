"""Shared helpers for Streamlit frontends (errors, env display, API responses)."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

MIN_DESCRIPTION_CHARS = 10

_GUARDRAIL_LABELS: dict[str, str] = {
    "moderation": "Moderación de contenido",
    "prompt_injection": "Manipulación de instrucciones",
    "pii": "Datos personales (PII)",
}


def env_display(key: str, default: str = "—") -> str:
    value = os.getenv(key, "").strip()
    return value if value else default


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
        if status == 422:
            return (
                "**Solicitud rechazada (422)** — la API no pudo validar el mensaje.\n\n"
                f"{detail}\n\n"
                f"_Consejo: la descripción debe tener al menos {MIN_DESCRIPTION_CHARS} "
                "caracteres (p. ej. un resumen de reunión, no solo un saludo)._"
            )
        if status == 415:
            return f"**Tipo de archivo no admitido (415)**\n\n{detail}"
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
