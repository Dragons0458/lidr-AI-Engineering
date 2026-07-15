"""Logfire bootstrap for Session 13 full-stack tracing.

Configured once at startup. Without a write token the SDK stays local
(``send_to_logfire='if-token-present'``) so offline tests and local boots work.
Spans must never carry transcripts, prompts, API keys, or raw documents.
"""

from __future__ import annotations

from typing import Any

import logfire
import structlog

log = structlog.get_logger()


def configure_logfire(
    *,
    token: str | None,
    service_name: str = "estimador-cag",
    environment: str = "development",
) -> None:
    """Configure Logfire once. Safe when the token is absent."""
    # Prefer explicit token when provided; otherwise honour env / skip send.
    kwargs: dict[str, Any] = {
        "service_name": service_name,
        "environment": environment,
        "send_to_logfire": "if-token-present",
    }
    if token:
        kwargs["token"] = token
    logfire.configure(**kwargs)
    log.info(
        "logfire_configured",
        send_enabled=bool(token),
        service_name=service_name,
    )


def instrument_fastapi_app(app: Any) -> None:
    """Attach FastAPI instrumentation after ``configure_logfire``."""
    try:
        logfire.instrument_fastapi(app)
    except Exception as exc:  # noqa: BLE001
        log.warning("logfire_fastapi_instrument_failed", error=str(exc)[:200])


def instrument_http_clients() -> None:
    """Instrument HTTPX (covers the OpenAI async client transport)."""
    try:
        logfire.instrument_httpx()
    except Exception as exc:  # noqa: BLE001
        log.warning("logfire_httpx_instrument_failed", error=str(exc)[:200])


def instrument_asyncpg() -> None:
    """Instrument asyncpg when the helper is available in this Logfire version."""
    instrument = getattr(logfire, "instrument_asyncpg", None)
    if instrument is None:
        return
    try:
        instrument()
    except Exception as exc:  # noqa: BLE001
        log.warning("logfire_asyncpg_instrument_failed", error=str(exc)[:200])
