"""API-key authentication for service-to-service calls.

Session 9 routers keep dedicated keys (``ESTIMATE_API_KEY`` /
``RETRIEVAL_API_KEY``). Session 15 adds a global gate
(``require_service_token``) so previously unprotected routers also demand a
token once the compose frontier closes the host ports.

``AI_SERVICE_TOKEN`` is an alias of ``ESTIMATE_API_KEY`` (exercise naming).
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Request, status

from app.config import get_settings

_API_KEY_HEADER = "X-API-Key"

# Paths that stay public so HEALTHCHECKs and OpenAPI docs work without a token.
PUBLIC_PATH_PREFIXES: tuple[str, ...] = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
)


def _verify(provided: str | None, expected: str | None) -> None:
    """Raise 401 unless ``provided`` matches the configured ``expected`` key."""
    if not expected or not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": _API_KEY_HEADER},
        )


def _matches_any(provided: str | None, candidates: list[str]) -> bool:
    if not provided or not candidates:
        return False
    return any(secrets.compare_digest(provided, candidate) for candidate in candidates)


def is_public_path(path: str) -> bool:
    """Return True when ``path`` is allowlisted (exact or under a public prefix)."""
    normalised = path.rstrip("/") or "/"
    for prefix in PUBLIC_PATH_PREFIXES:
        if normalised == prefix or normalised.startswith(f"{prefix}/"):
            return True
        # ``/health`` allowlist also covers ``/health/ready``.
        if prefix == "/health" and normalised.startswith("/health"):
            return True
    return False


def _service_token_from(settings: object) -> str | None:
    """Resolve ESTIMATE_API_KEY / AI_SERVICE_TOKEN from Settings or test stubs."""
    token = getattr(settings, "effective_service_token", None)
    if token:
        return token
    return getattr(settings, "AI_SERVICE_TOKEN", None) or getattr(
        settings, "ESTIMATE_API_KEY", None
    )


async def require_retrieval_key(
    x_api_key: str | None = Header(default=None, alias=_API_KEY_HEADER),
) -> None:
    """FastAPI dependency guarding ``POST /v1/retrieval/search``."""
    _verify(x_api_key, get_settings().RETRIEVAL_API_KEY)


async def require_estimate_key(
    x_api_key: str | None = Header(default=None, alias=_API_KEY_HEADER),
) -> None:
    """FastAPI dependency guarding estimate / agent / graph / supervisor routers."""
    _verify(x_api_key, _service_token_from(get_settings()))


async def require_service_token(
    request: Request,
    x_api_key: str | None = Header(default=None, alias=_API_KEY_HEADER),
) -> None:
    """Global gate: every non-public route needs a configured service token.

    Accepts ``AI_SERVICE_TOKEN`` / ``ESTIMATE_API_KEY`` (same effective value) or
    ``RETRIEVAL_API_KEY`` so Streamlit can keep sending the retrieval key on
    search calls. Router-level deps still enforce their specific key.
    """
    if is_public_path(request.url.path):
        return

    settings = get_settings()
    candidates = [
        token
        for token in (
            _service_token_from(settings),
            getattr(settings, "RETRIEVAL_API_KEY", None),
        )
        if token
    ]
    if not candidates:
        # No token configured → fail closed (matches Session 9 blank-key behaviour).
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": _API_KEY_HEADER},
        )
    if not _matches_any(x_api_key, candidates):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": _API_KEY_HEADER},
        )
