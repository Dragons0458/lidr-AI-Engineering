"""Agent-level sandboxing (Session 14 LIVE): three containment layers for WRITES.

Layer 1 — GRANTS with a RISK dimension (``AGENT_TOOL_GRANTS`` + ``TOOL_RISK``).
Layer 2 — Argument validation + tenancy before execution (``guard_action``).
Layer 3 — Audit of every intent with effects, including denied/deferred.

No process isolation here (Session 15). The default sink records intent in-process.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import structlog

from app.generation.agentic.graph.supervisor_privilege import (
    AGENT_PRIVILEGES,
    _digest,
    redact_args,
)

log = structlog.get_logger()

SAVE_ESTIMATE_TOOL = "save_estimate"


class ToolRisk(enum.StrEnum):
    """How much damage a tool can do if the model calls it wrongly."""

    PURE = "pure"
    READ = "read"
    WRITE = "write"
    IRREVERSIBLE = "irreversible"


TOOL_RISK: dict[str, ToolRisk] = {
    "search_budgets": ToolRisk.READ,
    "calculate_estimate": ToolRisk.PURE,
    "validate_estimate": ToolRisk.READ,
    SAVE_ESTIMATE_TOOL: ToolRisk.IRREVERSIBLE,
}


AGENT_TOOL_GRANTS: dict[str, frozenset[str]] = {
    **AGENT_PRIVILEGES,
    "persistence_agent": frozenset({SAVE_ESTIMATE_TOOL}),
}


class GrantVerificationError(RuntimeError):
    """A tool grant references a tool with no declared risk or unknown tool."""


def verify_tool_grants(known_tools: set[str] | None = None) -> None:
    """Fail fast if the grant table is inconsistent. Called at graph-build time."""
    known = known_tools if known_tools is not None else set(TOOL_RISK)
    for agent, tools in AGENT_TOOL_GRANTS.items():
        for tool in tools:
            if tool not in TOOL_RISK:
                raise GrantVerificationError(
                    f"agent {agent!r} is granted tool {tool!r}, which has no declared "
                    f"ToolRisk — every granted tool must be classified in TOOL_RISK"
                )
            if tool not in known:
                raise GrantVerificationError(
                    f"agent {agent!r} is granted tool {tool!r}, which is not a known "
                    f"tool ({sorted(known)})"
                )


@dataclass(frozen=True)
class ActionRequest:
    """A request to perform a tool action, carrying the run it belongs to."""

    agent: str
    tool: str
    args: dict[str, Any]
    estimation_id: str | None
    step: int


@dataclass(frozen=True)
class GuardDecision:
    """Guard verdict. Irreversible without approval → allowed but requires_human_approval."""

    allowed: bool
    requires_human_approval: bool = False
    reason: str = ""
    risk: ToolRisk | None = None
    redacted_args: dict[str, Any] = field(default_factory=dict)


def _human_approved(state: dict[str, Any]) -> bool:
    decision = state.get("human_decision") or {}
    action = decision.get("decision") or decision.get("action")
    return action == "approve"


def guard_action(req: ActionRequest, state: dict[str, Any]) -> GuardDecision:
    """Decide whether ``req`` may execute — privilege + args + tenancy, all pure."""
    redacted = redact_args(req.args if isinstance(req.args, dict) else {})
    risk = TOOL_RISK.get(req.tool)

    granted = AGENT_TOOL_GRANTS.get(req.agent, frozenset())
    if req.tool not in granted:
        return GuardDecision(
            allowed=False,
            reason=(
                f"agent {req.agent!r} is not granted tool {req.tool!r} "
                f"(granted: {sorted(granted) or 'none'})"
            ),
            risk=risk,
            redacted_args=redacted,
        )

    if not isinstance(req.args, dict):
        return GuardDecision(
            allowed=False,
            reason="arguments must be an object",
            risk=risk,
            redacted_args=redacted,
        )

    if req.tool == SAVE_ESTIMATE_TOOL and not req.args.get("estimate"):
        return GuardDecision(
            allowed=False,
            reason="save_estimate requires a non-empty 'estimate' payload",
            risk=risk,
            redacted_args=redacted,
        )

    run_id = state.get("estimation_id")
    if req.estimation_id != run_id:
        return GuardDecision(
            allowed=False,
            reason=(
                f"action estimation_id {req.estimation_id!r} does not match the "
                f"current run {run_id!r}"
            ),
            risk=risk,
            redacted_args=redacted,
        )
    args_id = req.args.get("estimation_id")
    if args_id is not None and args_id != run_id:
        return GuardDecision(
            allowed=False,
            reason=(
                f"argument estimation_id {args_id!r} does not match the "
                f"current run {run_id!r}"
            ),
            risk=risk,
            redacted_args=redacted,
        )

    if risk == ToolRisk.IRREVERSIBLE and not _human_approved(state):
        return GuardDecision(
            allowed=True,
            requires_human_approval=True,
            reason=(
                "irreversible action requires a human approval; route through the gate"
            ),
            risk=risk,
            redacted_args=redacted,
        )

    return GuardDecision(allowed=True, reason="ok", risk=risk, redacted_args=redacted)


SaveSink = Callable[[str | None, dict[str, Any]], dict[str, Any]]

PERSISTED: dict[str, dict[str, Any]] = {}


def record_intent_sink(
    estimation_id: str | None, estimate: dict[str, Any]
) -> dict[str, Any]:
    """Default side-effect-free sink: record in-process and log the intent."""
    record = {"estimation_id": estimation_id, "estimate": estimate}
    PERSISTED[estimation_id or "?"] = record
    log.info(
        "persistence_would_write",
        estimation_id=estimation_id,
        total_hours=(estimate or {}).get("total_hours"),
    )
    return {"ok": True, "stored": True}


async def execute_guarded(
    req: ActionRequest,
    state: dict[str, Any],
    *,
    sink: SaveSink | None = None,
    audit_preview_chars: int = 200,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Guard, execute (if cleared), and audit — returns ``(envelope, contribution)``."""
    started = perf_counter()
    digest = _digest(req.args if isinstance(req.args, dict) else {})
    decision = guard_action(req, state)
    preview = str(decision.redacted_args)[:audit_preview_chars]

    def _contribution(outcome: str, summary: str) -> dict[str, Any]:
        return {
            "step": req.step,
            "agent": req.agent,
            "action": f"tool:{req.tool}",
            "tool": req.tool,
            "outcome": outcome,
            "summary": summary[:200],
            "args_digest": digest,
            "duration_ms": int((perf_counter() - started) * 1000),
        }

    if not decision.allowed:
        log.error(
            "agent_privilege_denied",
            estimation_id=req.estimation_id,
            step=req.step,
            agent=req.agent,
            tool=req.tool,
            risk=str(decision.risk),
            args_digest=digest,
            args_preview=preview,
            reason=decision.reason,
        )
        return (
            {"ok": False, "error": "denied", "summary": decision.reason},
            _contribution("denied", decision.reason),
        )

    if decision.requires_human_approval:
        log.warning(
            "agent_action_deferred",
            estimation_id=req.estimation_id,
            step=req.step,
            agent=req.agent,
            tool=req.tool,
            risk=str(decision.risk),
            args_digest=digest,
            args_preview=preview,
            reason=decision.reason,
        )
        return (
            {
                "ok": False,
                "error": "awaiting_human_approval",
                "summary": decision.reason,
            },
            _contribution("deferred", decision.reason),
        )

    try:
        run_sink = sink or record_intent_sink
        result = run_sink(req.estimation_id, req.args.get("estimate") or {})
        if "ok" not in result:
            result = {**result, "ok": True}
        outcome, summary = "ok", "estimate persisted (guarded, human-authorised)"
    except Exception as exc:  # noqa: BLE001 — a failed write must not kill the graph
        result = {
            "ok": False,
            "error": type(exc).__name__,
            "summary": str(exc)[:200],
        }
        outcome, summary = "error", str(exc)[:200]

    log.info(
        "agent_action",
        estimation_id=req.estimation_id,
        step=req.step,
        agent=req.agent,
        tool=req.tool,
        action=f"tool:{req.tool}",
        outcome=outcome,
        risk=str(decision.risk),
        args_digest=digest,
        args_preview=preview,
        result_summary=summary,
    )
    return result, _contribution(outcome, summary)
