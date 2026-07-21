"""Minimum privilege over tools + the audit trail (Session 14 Level 3).

Each agent may only call tools listed in ``AGENT_PRIVILEGES``. ``guarded_dispatch``
checks the allowlist BEFORE executing and returns ``(envelope, contribution)`` so
agents stay pure ``state → partial update`` functions.
"""

from __future__ import annotations

import hashlib
import json
from time import perf_counter
from typing import Any

import structlog

from app.generation.agentic.agent_tools import RetrievalBackend, dispatch_tool

log = structlog.get_logger()


AGENT_PRIVILEGES: dict[str, frozenset[str]] = {
    "supervisor": frozenset(),
    "requirements_extractor": frozenset(),
    "budget_searcher": frozenset({"search_budgets"}),
    "estimate_generator": frozenset({"calculate_estimate"}),
    "coherence_validator": frozenset({"validate_estimate"}),
}


class PrivilegeViolation(RuntimeError):
    """An agent attempted a tool outside its declared allowlist."""

    def __init__(self, agent: str, tool: str, allowed: frozenset[str]) -> None:
        self.agent = agent
        self.tool = tool
        self.allowed = allowed
        super().__init__(
            f"agent {agent!r} attempted tool {tool!r}; its declared privilege is "
            f"{sorted(allowed) or 'NO tools'}"
        )


def allowed_tools(agent: str) -> frozenset[str]:
    """The tools ``agent`` may call. An unknown agent has no privilege at all."""
    return AGENT_PRIVILEGES.get(agent, frozenset())


def assert_allowed(agent: str, tool: str) -> None:
    """Raise ``PrivilegeViolation`` unless ``tool`` is in ``agent``'s allowlist."""
    allowed = allowed_tools(agent)
    if tool not in allowed:
        raise PrivilegeViolation(agent, tool, allowed)


def _digest(args: dict[str, Any]) -> str:
    canonical = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _preview(args: dict[str, Any], *, limit: int) -> str:
    return json.dumps(args, sort_keys=True, default=str)[:limit]


async def _unused_backend(_args: Any) -> list[dict[str, Any]]:
    return []


def record_model_action(
    agent: str,
    action: str,
    *,
    step: int,
    summary: str,
    estimation_id: str | None = None,
    duration_ms: int | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Audit a model-only action (no tool) and return its contribution row."""
    log.info(
        "agent_action",
        estimation_id=estimation_id,
        step=step,
        agent=agent,
        tool=None,
        action=action,
        outcome="ok",
        allowed=sorted(allowed_tools(agent)),
        model=model,
        result_summary=summary[:200],
        duration_ms=duration_ms,
    )
    return {
        "step": step,
        "agent": agent,
        "action": action,
        "tool": None,
        "outcome": "ok",
        "summary": summary[:200],
        "args_digest": None,
        "duration_ms": duration_ms,
    }


async def guarded_dispatch(
    agent: str,
    tool: str,
    args: dict[str, Any],
    *,
    step: int,
    estimation_id: str | None = None,
    backend: RetrievalBackend | None = None,
    privilege_strict: bool = False,
    audit_preview_chars: int = 200,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check privilege, execute, audit. Returns ``(result_envelope, contribution)``."""
    started = perf_counter()
    digest = _digest(args)
    allowed = allowed_tools(agent)
    preview = _preview(args, limit=audit_preview_chars)

    if tool not in allowed:
        violation = PrivilegeViolation(agent, tool, allowed)
        log.error(
            "agent_privilege_denied",
            estimation_id=estimation_id,
            step=step,
            agent=agent,
            tool=tool,
            allowed=sorted(allowed),
            args_digest=digest,
            args_preview=preview,
        )
        contribution = {
            "step": step,
            "agent": agent,
            "action": f"tool:{tool}",
            "tool": tool,
            "outcome": "denied",
            "summary": str(violation),
            "args_digest": digest,
            "duration_ms": int((perf_counter() - started) * 1000),
        }
        if privilege_strict:
            raise violation
        return (
            {"ok": False, "error": "privilege_denied", "summary": str(violation)},
            contribution,
        )

    try:
        if tool == "search_budgets" and backend is None:
            raise ValueError("search_budgets requires a retrieval backend")
        result = await dispatch_tool(
            tool, args, backend=backend if backend is not None else _unused_backend
        )
        if "ok" not in result:
            result = {**result, "ok": True}
        outcome = "ok"
    except Exception as exc:  # noqa: BLE001 — soft-fail; never kill the graph
        result = {
            "ok": False,
            "error": type(exc).__name__,
            "summary": str(exc)[:200],
        }
        outcome = "error"

    duration_ms = int((perf_counter() - started) * 1000)
    summary = str(result.get("summary", ""))[:200]
    log.info(
        "agent_action",
        estimation_id=estimation_id,
        step=step,
        agent=agent,
        tool=tool,
        action=f"tool:{tool}",
        outcome=outcome,
        allowed=sorted(allowed),
        args_digest=digest,
        args_preview=preview,
        result_summary=summary,
        duration_ms=duration_ms,
    )
    return result, {
        "step": step,
        "agent": agent,
        "action": f"tool:{tool}",
        "tool": tool,
        "outcome": outcome,
        "summary": summary,
        "args_digest": digest,
        "duration_ms": duration_ms,
    }
