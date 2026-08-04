"""Streamlit helpers for the Session 14 supervisor multi-agent flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from streamlit_ui.common import get_api_root_url, get_estimate_api_key

SUPERVISOR_PREFIX = "/v1/estimate/agent/supervisor"

EXERCISES = Path(__file__).resolve().parent.parent / "exercises" / "session-14"

_EDGE_CASE_NAMES = frozenset(
    {"low_confidence", "no_precedent", "out_of_historical_range"}
)


def load_sample_transcript(name: str) -> str:
    """Load a sample transcript by short name or edge-case signal name."""
    if name in _EDGE_CASE_NAMES:
        path = EXERCISES / "edge_cases" / f"{name}.txt"
    else:
        path = EXERCISES / f"sample_transcript_{name}.txt"
    return path.read_text(encoding="utf-8")


def _headers(api_key: str | None = None) -> dict[str, str]:
    key = api_key or get_estimate_api_key()
    return {"X-API-Key": key, "Content-Type": "application/json"}


def supervisor_start(
    transcript: str,
    *,
    estimation_id: str | None = None,
    api_root: str | None = None,
    api_key: str | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    base = api_root or get_api_root_url()
    payload: dict[str, Any] = {"transcript": transcript}
    if estimation_id:
        payload["estimation_id"] = estimation_id
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{base}{SUPERVISOR_PREFIX}",
            json=payload,
            headers=_headers(api_key),
        )
        response.raise_for_status()
        return response.json()


def supervisor_resume(
    estimation_id: str,
    decision: str,
    *,
    estimate_overrides: dict | None = None,
    note: str | None = None,
    api_root: str | None = None,
    api_key: str | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    base = api_root or get_api_root_url()
    payload: dict[str, Any] = {"decision": decision}
    if estimate_overrides is not None:
        payload["estimate_overrides"] = estimate_overrides
    if note:
        payload["note"] = note
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{base}{SUPERVISOR_PREFIX}/{estimation_id}/resume",
            json=payload,
            headers=_headers(api_key),
        )
        response.raise_for_status()
        return response.json()


def supervisor_state(
    estimation_id: str,
    *,
    api_root: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    base = api_root or get_api_root_url()
    with httpx.Client(timeout=timeout) as client:
        response = client.get(
            f"{base}{SUPERVISOR_PREFIX}/{estimation_id}/state",
            headers=_headers(api_key),
        )
        response.raise_for_status()
        return response.json()


def status_badge_label(status: str | None) -> str:
    return {
        "awaiting_human_review": "⏳ Awaiting human review",
        "validated": "✅ Validated",
        # Soft flag from the coherence validator — NOT the HITL interrupt.
        "needs_review": "⚠ Completed with warnings",
        "rejected": "⛔ Rejected",
    }.get(status or "", status or "unknown")


def confidence_pct(value: float | None) -> int | None:
    """Nil-safe confidence as an integer percentage."""
    if value is None:
        return None
    try:
        return int(round(float(value) * 100))
    except (TypeError, ValueError):
        return None


def deferred_rows(contributions: list[dict] | None) -> list[dict]:
    return [row for row in (contributions or []) if row.get("outcome") == "deferred"]


def irreversible_rows(contributions: list[dict] | None) -> list[dict]:
    return [row for row in (contributions or []) if row.get("tool") == "save_estimate"]


def competition_summary(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Derive a compact competition block from divergence + synthesis / range."""
    if not state:
        return None
    divergence = state.get("divergence") or {}
    synthesis = state.get("synthesis") or {}
    estimate = state.get("estimate") or {}
    est_range = estimate.get("range") or {}
    low = synthesis.get("low")
    high = synthesis.get("high")
    if low is None:
        low = est_range.get("low")
    if high is None:
        high = est_range.get("high")
    if low is None and high is None and not divergence:
        return None
    return {
        "low": low,
        "high": high,
        "spread": divergence.get("spread"),
        "ratio": divergence.get("ratio"),
        "level": divergence.get("level"),
        "open_questions": list(
            synthesis.get("open_questions") or estimate.get("open_questions") or []
        ),
    }
