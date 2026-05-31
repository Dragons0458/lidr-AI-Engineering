"""Input guardrails: moderation, prompt-injection patterns, and PII detection."""

from __future__ import annotations

import re
from typing import Literal

import litellm
import structlog

from app.config import get_settings

log = structlog.get_logger()

Reason = Literal["moderation", "prompt_injection", "pii"]

_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore (previous|prior|all|the) (instructions|prompts|rules)",
        r"</?(system|instructions|prompt)>",
        r"new instructions:",
        r"forget (everything|all|previous)",
        r"you are now",
        r"disregard .{0,80} (instructions|rules|context|previous)",
    )
)

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    re.IGNORECASE,
)
_IBAN_RE = re.compile(
    r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\d)",
)


class InputGuardrailViolation(Exception):
    """Raised when input fails a guardrail check."""

    def __init__(self, *, reason: Reason, message: str) -> None:
        self.reason = reason
        self.message = message
        super().__init__(message)


def check_input(
    description: str,
    *,
    attachments_text: list[str] | None = None,
    run_moderation: bool = True,
) -> None:
    """Validate combined description and attachment text; raise on first violation."""
    parts = [description, *(attachments_text or [])]
    text = "\n".join(part for part in parts if part).strip()
    if not text:
        return

    if run_moderation:
        _check_moderation(text)
    _check_prompt_injection(text)
    _check_pii(text)


def _check_moderation(text: str) -> None:
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        return

    try:
        response = litellm.moderation(model="text-moderation-latest", input=text)
    except Exception as exc:
        log.warning("moderation_check_failed_open", error=str(exc))
        return

    results = _moderation_results(response)
    if not results:
        return

    first = results[0]
    flagged = _moderation_flagged(first)
    if not flagged:
        return

    categories = _moderation_categories(first)
    flagged_names = [name for name, active in categories.items() if active]
    detail = ", ".join(flagged_names) if flagged_names else "policy violation"
    raise InputGuardrailViolation(
        reason="moderation",
        message=f"Content flagged by moderation: {detail}",
    )


def _check_prompt_injection(text: str) -> None:
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            raise InputGuardrailViolation(
                reason="prompt_injection",
                message="Potential prompt injection detected in input",
            )


def _check_pii(text: str) -> None:
    if _EMAIL_RE.search(text):
        raise InputGuardrailViolation(
            reason="pii",
            message="Email address detected in input",
        )
    if _IBAN_RE.search(text):
        raise InputGuardrailViolation(
            reason="pii",
            message="IBAN detected in input",
        )
    if _PHONE_RE.search(text):
        raise InputGuardrailViolation(
            reason="pii",
            message="Phone number detected in input",
        )


def _moderation_results(response: object) -> list[object]:
    if isinstance(response, dict):
        return list(response.get("results") or [])
    results = getattr(response, "results", None)
    return list(results) if results else []


def _moderation_flagged(result: object) -> bool:
    if isinstance(result, dict):
        return bool(result.get("flagged"))
    return bool(getattr(result, "flagged", False))


def _moderation_categories(result: object) -> dict[str, bool]:
    if isinstance(result, dict):
        categories = result.get("categories") or {}
    else:
        categories = getattr(result, "categories", None) or {}

    if isinstance(categories, dict):
        return {str(key): bool(value) for key, value in categories.items()}

    return {
        str(key): bool(getattr(categories, key, False))
        for key in dir(categories)
        if not key.startswith("_")
    }
