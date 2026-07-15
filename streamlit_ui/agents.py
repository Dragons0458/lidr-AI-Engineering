"""Local agent profiles and presentation helpers for the Streamlit UI."""

from __future__ import annotations

import base64
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

AGENT_TYPE = "handwritten"
INHERIT_LABEL = "Por defecto del servicio"
ALLOWED_EFFORTS = {"minimal", "low", "medium", "high"}
DEFAULT_AVATAR_MAX_BYTES = 2_097_152
DEFAULT_HOURLY_RATE_EUR = 75.0
STATIC_AGENT_DEFAULTS: dict[str, Any] = {
    "model": "gpt-5",
    "reasoning_effort": "medium",
    "max_iterations": 10,
    "search_top_k": 5,
    "search_distance_threshold": 0.45,
}


def _positive_env_float(name: str, fallback: float) -> float:
    try:
        value = float(os.getenv(name, str(fallback)))
    except ValueError:
        return fallback
    return value if value > 0 else fallback


def _positive_env_int(name: str, fallback: int) -> int:
    try:
        value = int(os.getenv(name, str(fallback)))
    except ValueError:
        return fallback
    return value if value > 0 else fallback


STREAMLIT_DEFAULT_HOURLY_RATE_EUR = _positive_env_float(
    "STREAMLIT_DEFAULT_HOURLY_RATE_EUR", DEFAULT_HOURLY_RATE_EUR
)
STREAMLIT_AGENT_AVATAR_MAX_BYTES = _positive_env_int(
    "STREAMLIT_AGENT_AVATAR_MAX_BYTES", DEFAULT_AVATAR_MAX_BYTES
)


def compact_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Remove inherited/empty values while preserving valid zero-like values."""
    return {
        key: value
        for key, value in (config or {}).items()
        if value is not None and (not isinstance(value, str) or value.strip())
    }


@dataclass(slots=True)
class AgentProfile:
    """A FastAPI-independent profile persisted exclusively by Streamlit."""

    name: str
    persona: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    id: int | None = None
    agent_type: str = AGENT_TYPE
    is_default: bool = False
    avatar_filename: str | None = None
    avatar_content_type: str | None = None
    avatar_bytes: bytes | None = None
    created_at: str | datetime | None = None
    updated_at: str | datetime | None = None

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.persona = self.persona.strip()
        self.agent_type = self.agent_type.strip() or AGENT_TYPE
        self.config = compact_config(self.config)
        if not self.name:
            raise ValueError("Profile name cannot be empty.")
        if len(self.name) > 120:
            raise ValueError("Profile name cannot exceed 120 characters.")
        if len(self.persona) > 2000:
            raise ValueError("Persona cannot exceed 2000 characters.")
        effort = self.config.get("reasoning_effort")
        if effort is not None and effort not in ALLOWED_EFFORTS:
            raise ValueError("Invalid reasoning effort.")
        self._validate_range("max_iterations", 1, 20)
        self._validate_range("search_top_k", 1, 30)
        self._validate_range("search_distance_threshold", 0, 2)
        if self.avatar_bytes is not None:
            detected = validate_avatar(self.avatar_bytes, self.avatar_content_type)
            self.avatar_content_type = detected

    def _validate_range(self, key: str, minimum: float, maximum: float) -> None:
        value = self.config.get(key)
        if value is not None and not minimum <= float(value) <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}.")

    def phase_payload(self, phase: Literal["structure", "hours"]) -> dict[str, Any]:
        return phase_payload(self, phase)

    def snapshot(self, *, include_avatar: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not include_avatar:
            payload.pop("avatar_bytes", None)
        for key in ("created_at", "updated_at"):
            if isinstance(payload.get(key), datetime):
                payload[key] = payload[key].isoformat()
        return payload


def phase_payload(
    profile: AgentProfile | dict[str, Any] | None,
    phase: Literal["structure", "hours"],
) -> dict[str, Any]:
    """Build a phase-specific request fragment without inherited values."""
    if profile is None:
        return {}
    if isinstance(profile, AgentProfile):
        config = profile.config
        persona = profile.persona
    else:
        config = compact_config(
            profile.get("config") or profile.get("config_payload") or {}
        )
        persona = str(profile.get("persona") or "").strip()
    keys = ["model", "reasoning_effort"]
    if phase == "hours":
        keys.extend(["max_iterations", "search_top_k", "search_distance_threshold"])
    payload = {key: config.get(key) for key in keys}
    payload["persona"] = persona or None
    return compact_config(payload)


PRESET_PROFILES = (
    AgentProfile(
        name="Estándar",
        persona=(
            "Busca análogos históricos por cada tarea marcada, reformula la búsqueda "
            "cuando sea necesario y conserva únicamente derivaciones respaldadas."
        ),
        config={"model": "gpt-5", "reasoning_effort": "medium"},
        is_default=True,
    ),
    AgentProfile(
        name="Veloz (debug)",
        persona=(
            "Prioriza búsquedas breves por tarea, usa evidencia cercana y evita "
            "iteraciones que no aporten nuevos análogos."
        ),
        config={
            "model": "gpt-5-mini",
            "reasoning_effort": "low",
            "max_iterations": 6,
        },
    ),
    AgentProfile(
        name="Exhaustivo",
        persona=(
            "Investiga cada tarea marcada con varias reformulaciones, contrasta "
            "análogos y deriva horas solo cuando la evidencia converja."
        ),
        config={
            "model": "gpt-5",
            "reasoning_effort": "high",
            "max_iterations": 15,
            "search_top_k": 8,
        },
    ),
)


def validate_avatar(
    data: bytes,
    declared_content_type: str | None = None,
    *,
    max_bytes: int | None = None,
) -> str:
    """Return the detected MIME type after validating file signature and size."""
    limit = max_bytes or STREAMLIT_AGENT_AVATAR_MAX_BYTES
    if not data:
        raise ValueError("Avatar cannot be empty.")
    if len(data) > limit:
        raise ValueError(f"Avatar exceeds the {limit} byte limit.")
    detected: str | None = None
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        detected = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        detected = "image/jpeg"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        detected = "image/gif"
    elif len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        detected = "image/webp"
    if detected is None:
        raise ValueError("Avatar must be a PNG, JPEG, GIF, or WEBP image.")
    if declared_content_type and declared_content_type.lower() != detected:
        raise ValueError("Avatar content type does not match its binary signature.")
    return detected


detect_avatar_content_type = validate_avatar
validate_avatar_upload = validate_avatar


def avatar_data_uri(profile: AgentProfile | dict[str, Any] | None) -> str | None:
    if profile is None:
        return None
    if isinstance(profile, AgentProfile):
        content, content_type = profile.avatar_bytes, profile.avatar_content_type
    else:
        content = profile.get("avatar_bytes")
        content_type = profile.get("avatar_content_type")
    if not content or not content_type:
        return None
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def profile_summary(profile: AgentProfile | dict[str, Any] | None) -> str:
    if profile is None:
        return INHERIT_LABEL
    config = (
        profile.config
        if isinstance(profile, AgentProfile)
        else (profile.get("config") or profile.get("config_payload") or {})
    )
    labels = {
        "model": "modelo",
        "reasoning_effort": "effort",
        "max_iterations": "iteraciones",
        "search_top_k": "top-k",
        "search_distance_threshold": "distancia",
    }
    return " · ".join(
        f"{labels[key]}: {config.get(key, f'hereda ({value})')}"
        for key, value in STATIC_AGENT_DEFAULTS.items()
    )


def profile_defaults(profile: AgentProfile | dict[str, Any] | None) -> dict[str, Any]:
    """Resolve documented static defaults for display/testing, not runtime claims."""
    if profile is None:
        return dict(STATIC_AGENT_DEFAULTS)
    config = (
        profile.config
        if isinstance(profile, AgentProfile)
        else (profile.get("config") or profile.get("config_payload") or {})
    )
    return {**STATIC_AGENT_DEFAULTS, **compact_config(config)}


def normalize_agent_trace(trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, step in enumerate((trace or {}).get("steps") or [], start=1):
        normalized.append(
            {
                "step": int(step.get("step") or index),
                "reasoning_summary": step.get("reasoning_summary")
                or "(sin resumen de razonamiento)",
                "tool": str(step.get("tool") or ""),
                "tool_args": dict(step.get("tool_args") or {}),
                "observation": str(step.get("observation") or ""),
            }
        )
    return normalized


def trace_summary(trace: dict[str, Any] | None) -> str:
    steps = (trace or {}).get("steps") or []
    searches = sum(step.get("tool") == "search_budgets" for step in steps)
    derives = sum(step.get("tool") == "derive_task_hours" for step in steps)
    return f"{len(steps)} pasos · {searches} búsquedas · {derives} derivaciones"


def get_default_hourly_rate_eur() -> float:
    return STREAMLIT_DEFAULT_HOURLY_RATE_EUR
