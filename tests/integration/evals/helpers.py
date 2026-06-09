import re


def estimate_real(transcript: str):
    from app.foundation.formatters import format_response
    from app.domain.schemas.estimation import (
        DetailLevel,
        EstimationRequest,
        OutputFormat,
        ProjectType,
    )
    from app.domain.estimation_service import generate_estimation

    request = EstimationRequest(
        description=transcript,
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.LINE_ITEMS,
    )
    return format_response(generate_estimation(request, prompt_version="v1"))


def extract_total_hours(estimation: str) -> int:
    normalized = estimation.lower()
    total_patterns = [
        r"total(?:\s+\w+){0,5}\s*[:|-]?\s*(\d{1,5})\s*(?:h|horas|hours)\b",
        r"(\d{1,5})\s*(?:h|horas|hours)\b(?:\s+\w+){0,5}\s+total",
    ]
    for pattern in total_patterns:
        matches = [int(match) for match in re.findall(pattern, normalized)]
        if matches:
            return max(matches)

    hour_mentions = [
        int(match)
        for match in re.findall(r"\b(\d{1,5})\s*(?:h|horas|hours)\b", normalized)
    ]
    if not hour_mentions:
        raise AssertionError(
            f"Could not extract total hours from output:\n{estimation}"
        )
    return max(hour_mentions)


def missing_expected_components(
    estimation: str,
    expected_components: list[str],
) -> list[str]:
    normalized = estimation.lower()
    return [
        component
        for component in expected_components
        if not any(keyword in normalized for keyword in _COMPONENT_KEYWORDS[component])
    ]


_COMPONENT_KEYWORDS = {
    "frontend": [
        "frontend",
        "front-end",
        "interfaz",
        "ui",
        "landing",
        "dashboard",
        "portal",
    ],
    "form_handling": ["form", "formulario", "contacto", "lead"],
    "backend": ["backend", "back-end", "api", "servidor", "admin"],
    "auth": ["auth", "autentic", "login", "roles", "permisos", "permissions"],
    "audit_log": ["audit", "auditor", "log"],
    "email_jobs": ["email", "correo", "notific", "semanal", "weekly"],
    "reporting": ["report", "reporte", "dashboard", "informe"],
    "database": ["database", "base de datos", "datos", "persistencia"],
}
