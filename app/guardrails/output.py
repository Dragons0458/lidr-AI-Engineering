"""Output guardrails: out-of-scope detection and PII redaction (filter policy)."""

from __future__ import annotations

from dataclasses import dataclass

from app.guardrails.input import _EMAIL_RE, _IBAN_RE, _PHONE_RE

OUT_OF_SCOPE_PREFIX = "Out of scope:"


@dataclass(frozen=True)
class OutputGuardrailResult:
    text: str
    out_of_scope: bool
    pii_redacted: bool


def enforce_output(text: str) -> OutputGuardrailResult:
    """Apply scope and PII filters without raising."""
    stripped = text.strip()
    if stripped.startswith(OUT_OF_SCOPE_PREFIX):
        first_line = stripped.splitlines()[0].strip()
        return OutputGuardrailResult(
            text=first_line,
            out_of_scope=True,
            pii_redacted=False,
        )

    redacted, changed = _redact_pii(text)
    return OutputGuardrailResult(
        text=redacted,
        out_of_scope=False,
        pii_redacted=changed,
    )


def _redact_pii(text: str) -> tuple[str, bool]:
    redacted = _EMAIL_RE.sub("[REDACTED]", text)
    redacted = _IBAN_RE.sub("[REDACTED]", redacted)
    redacted = _PHONE_RE.sub("[REDACTED]", redacted)
    return redacted, redacted != text
