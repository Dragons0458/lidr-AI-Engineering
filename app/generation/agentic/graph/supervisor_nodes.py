"""Supervisor + four specialists + human gate — closed over ``SupervisorDeps``.

Unlike the course reference (module-level service locators), every I/O collaborator
is injected. Nodes stay pure ``state → partial update`` (or ``Command``) and unit
tests swap fakes without monkeypatching singletons.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import logfire
import structlog
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from app.generation.agentic.agent_schemas import (
    AgentComponent,
    AgentEstimate,
)
from app.generation.agentic.graph.nodes import (
    LegacyRetrievalBackend,
    ReformulateFn,
    StructureFn,
    requirements_from_brief,
    structure_to_components,
)
from app.generation.agentic.graph.supervisor_privilege import (
    guarded_dispatch,
    record_model_action,
)
from app.generation.agentic.graph.supervisor_state import (
    SupervisorState,
    apply_human_decision,
    build_state_digest,
    compute_confidence,
    detect_review_risks,
    estimate_for_historical_band,
    is_outside_historical_band,
    precedent_matches,
    requires_human_review,
)
from app.generation.agentic.graph.state import BudgetMatch
from app.generation.rag.task_hours import compose_task_search_text

log = structlog.get_logger()

SupervisorTarget = Literal[
    "requirements_extractor",
    "budget_searcher",
    "estimate_generator",
    "coherence_validator",
    "finish",
]

_ORDER: list[str] = [
    "requirements_extractor",
    "budget_searcher",
    "estimate_generator",
    "coherence_validator",
]


class SupervisorDecision(BaseModel):
    """Structured router output — the model cannot invent destinations."""

    next_agent: SupervisorTarget
    reason: str = Field(min_length=1)
    confidence: Literal["low", "medium", "high"] = "medium"


RouteFn = Callable[[str], Awaitable[SupervisorDecision]]


@dataclass(frozen=True)
class SupervisorDeps:
    """Injectable collaborators for the six supervisor-graph callables."""

    reformulate: ReformulateFn
    propose_structure: StructureFn
    retrieval_backend: LegacyRetrievalBackend
    route_with_model: RouteFn
    confidence_threshold: float
    min_grounded_ratio: float
    out_of_range_factor: float
    max_steps: int
    privilege_strict: bool
    audit_preview_chars: int = 200
    grounding_max_distance: float = 0.45


def _step_of(state: SupervisorState) -> int:
    return int(state.get("supervisor_steps") or 0)


def _already_ran(agent: str, state: SupervisorState) -> bool:
    """Whether ``agent`` was already dispatched (routing history, not output fullness)."""
    if agent == "budget_searcher" and state.get("search_completed"):
        return True
    return any(
        record.get("next_agent") == agent
        for record in (state.get("routing_history") or [])
    )


def _inputs_ready(agent: str, state: SupervisorState) -> bool:
    if agent == "requirements_extractor":
        return bool(state.get("transcript"))
    if agent == "budget_searcher":
        return bool(state.get("components"))
    if agent == "estimate_generator":
        # Search completed (even with empty matches) is enough — the gate catches
        # ungrounded estimates.
        return bool(state.get("components")) and (
            bool(state.get("search_completed"))
            or _already_ran("budget_searcher", state)
        )
    if agent == "coherence_validator":
        return bool(state.get("estimate"))
    return False


def _is_legal(target: str, state: SupervisorState) -> bool:
    if target == "finish":
        return True
    if target not in _ORDER:
        return False
    return _inputs_ready(target, state) and not _already_ran(target, state)


def _fallback_next(state: SupervisorState) -> str:
    for agent in _ORDER:
        if _is_legal(agent, state):
            return agent
    return "finish"


def _confidence_label(budgeted: int, total: int) -> str:
    if total == 0:
        return "low"
    ratio = budgeted / total
    if ratio >= 0.85:
        return "high"
    if ratio >= 0.5:
        return "medium"
    return "low"


def make_supervisor_nodes(deps: SupervisorDeps) -> dict[str, Callable[..., Any]]:
    """Build the six node callables closed over ``deps``."""

    async def supervisor(state: SupervisorState) -> Command:
        step = _step_of(state)
        decision_confidence: str | None = None

        if step >= deps.max_steps:
            target, reason, source = (
                "finish",
                f"step budget of {deps.max_steps} exhausted; finishing",
                "limit",
            )
            log.warning("supervisor_step_budget_exhausted", step=step)
        else:
            with logfire.span("supervisor: route"):
                digest = build_state_digest(
                    state,
                    grounding_max_distance=deps.grounding_max_distance,
                )
                try:
                    decision = await deps.route_with_model(digest)
                    target, reason, source = (
                        decision.next_agent,
                        decision.reason,
                        "llm",
                    )
                    decision_confidence = decision.confidence
                except Exception as exc:  # noqa: BLE001
                    target = _fallback_next(state)
                    reason = (
                        f"router unavailable ({type(exc).__name__}); "
                        "fell back to the dependency ladder"
                    )
                    source = "fallback"
                    log.error(
                        "supervisor_route_failed",
                        error=str(exc)[:200],
                        step=step,
                    )

                if not _is_legal(target, state):
                    overridden, target = target, _fallback_next(state)
                    log.warning(
                        "supervisor_route_overridden",
                        step=step,
                        proposed=overridden,
                        chosen=target,
                    )
                    reason = (
                        f"router proposed {overridden!r}, which is not legal "
                        f"in this state; overridden to {target!r}"
                    )
                    source = "fallback"

        goto = "human_review_gate" if target == "finish" else target
        log.info(
            "supervisor_route",
            step=step,
            next_agent=target,
            goto=goto,
            source=source,
            reason=reason[:200],
        )
        return Command(
            goto=goto,
            update={
                "next_agent": target,
                "route_reason": reason,
                "supervisor_steps": step + 1,
                "routing_history": [
                    {
                        "step": step,
                        "next_agent": target,
                        "reason": reason,
                        "source": source,
                        "decision_confidence": decision_confidence,
                    }
                ],
            },
        )

    async def requirements_extractor(state: SupervisorState) -> dict[str, Any]:
        with logfire.span("agent: requirements_extractor"):
            step = _step_of(state)
            estimation_id = state.get("estimation_id")
            contributions: list[dict] = []

            started = perf_counter()
            brief = await deps.reformulate(state["transcript"])
            requirements = requirements_from_brief(brief)
            contributions.append(
                record_model_action(
                    "requirements_extractor",
                    "extract_requirements",
                    step=step,
                    estimation_id=estimation_id,
                    summary=f"{len(requirements)} requirements extracted",
                    duration_ms=int((perf_counter() - started) * 1000),
                )
            )

            started = perf_counter()
            structure = await deps.propose_structure(brief)
            components = structure_to_components(structure)
            contributions.append(
                record_model_action(
                    "requirements_extractor",
                    "propose_structure",
                    step=step,
                    estimation_id=estimation_id,
                    summary=f"{len(components)} components classified",
                    duration_ms=int((perf_counter() - started) * 1000),
                )
            )

            return {
                "project_brief": brief.model_dump(mode="json"),
                "requirements": requirements,
                "components": components,
                "agent_contributions": contributions,
            }

    async def budget_searcher(state: SupervisorState) -> dict[str, Any]:
        with logfire.span("agent: budget_searcher"):
            step = _step_of(state)
            estimation_id = state.get("estimation_id")

            matches: list[BudgetMatch] = []
            contributions: list[dict] = []
            errors: list[str] = []

            for component in state.get("components") or []:
                query = compose_task_search_text(
                    component["category"],
                    component["name"],
                    component.get("description") or None,
                )
                # Do NOT hard-filter by sector: reformulated sectors rarely match the
                # corpus vocabulary exactly, and wipe recall on otherwise ordinary
                # happy-path briefs (portal / ERP / auth).
                result, contribution = await guarded_dispatch(
                    "budget_searcher",
                    "search_budgets",
                    {
                        "query": query,
                        "filters": {
                            "sectors": None,
                            "component_type": component["category"],
                        },
                    },
                    step=step,
                    estimation_id=estimation_id,
                    backend=deps.retrieval_backend,
                    privilege_strict=deps.privilege_strict,
                    audit_preview_chars=deps.audit_preview_chars,
                )
                contributions.append(contribution)

                if not result.get("ok", True) and result.get("error"):
                    errors.append(
                        f"budget search failed for {component['name']!r}: "
                        f"{result.get('summary')}"
                    )
                    continue

                for item in result.get("items") or []:
                    hours = item.get("estimated_hours")
                    if hours is None:
                        continue
                    matches.append(
                        BudgetMatch(
                            component_id=component["component_id"],
                            chunk_id=int(item["id"]),
                            reference_budget_id=item.get("budget_id"),
                            amount=float(hours),
                            distance=float(item.get("distance") or 0.0),
                        )
                    )

            update: dict[str, Any] = {
                "budget_matches": matches,
                "search_completed": True,
                "agent_contributions": contributions,
            }
            if errors:
                update["errors"] = errors
            return update

    async def estimate_generator(state: SupervisorState) -> dict[str, Any]:
        with logfire.span("agent: estimate_generator"):
            step = _step_of(state)
            estimation_id = state.get("estimation_id")
            components = state.get("components") or []
            by_id: dict[str, list[BudgetMatch]] = defaultdict(list)
            for match in state.get("budget_matches") or []:
                by_id[match["component_id"]].append(match)

            calc_components = [
                {
                    "name": component["component_id"],
                    "reference_amounts": [
                        m["amount"] for m in by_id[component["component_id"]]
                    ],
                }
                for component in components
            ]
            result, contribution = await guarded_dispatch(
                "estimate_generator",
                "calculate_estimate",
                {"components": calc_components},
                step=step,
                estimation_id=estimation_id,
                privilege_strict=deps.privilege_strict,
                audit_preview_chars=deps.audit_preview_chars,
            )

            if not result.get("ok", True) and result.get("error"):
                return {
                    "errors": [f"calculate_estimate failed: {result.get('summary')}"],
                    "agent_contributions": [contribution],
                }

            calc_rows = result.get("components") or []
            estimate_components: list[AgentComponent] = []
            assumptions: list[str] = []
            anchors: list[dict] = []
            budgeted = 0

            for component, row in zip(components, calc_rows, strict=False):
                matches = by_id[component["component_id"]]
                cited = [m["chunk_id"] for m in matches]
                hours = float(row.get("estimated_hours") or 0.0)
                unbudgeted = bool(row.get("unbudgeted"))
                anchors.append(
                    {
                        "component_id": component["component_id"],
                        "name": component["name"],
                        "estimated_hours": hours,
                        "unbudgeted": unbudgeted,
                        "reference_count": int(row.get("reference_count") or 0),
                    }
                )
                if unbudgeted:
                    assumptions.append(
                        f"No historical references for "
                        f"{component['category']}/{component['name']}; hours set to 0."
                    )
                    rationale = (
                        "No historical references found; left at 0h pending review."
                    )
                else:
                    budgeted += 1
                    rationale = (
                        f"Median of {row.get('reference_count', 0)} historical "
                        f"hour(s) plus 15% contingency for "
                        f"{component['category']}/{component['name']}."
                    )
                estimate_components.append(
                    AgentComponent(
                        name=component["name"],
                        estimated_hours=hours,
                        cited_chunk_ids=cited,
                        rationale=rationale,
                    )
                )

            draft = AgentEstimate(
                components=estimate_components,
                total_hours=float(result.get("total_hours") or 0.0),
                assumptions=assumptions,
                confidence=_confidence_label(budgeted, len(components)),
            )
            return {
                "estimate": draft.model_dump(mode="json"),
                "component_anchors": anchors,
                "agent_contributions": [contribution],
            }

    async def coherence_validator(state: SupervisorState) -> dict[str, Any]:
        with logfire.span("agent: coherence_validator"):
            step = _step_of(state)
            estimation_id = state.get("estimation_id")
            estimate = state.get("estimate") or {}
            matches = state.get("budget_matches") or []
            components = state.get("components") or []
            by_id: dict[str, list[BudgetMatch]] = defaultdict(list)
            for match in matches:
                by_id[match["component_id"]].append(match)

            validate_rows = []
            for component, estimated in zip(
                components, estimate.get("components") or [], strict=False
            ):
                validate_rows.append(
                    {
                        "name": component["name"],
                        "estimated_hours": float(
                            estimated.get("estimated_hours") or 0.0
                        ),
                        "reference_amounts": [
                            m["amount"] for m in by_id[component["component_id"]]
                        ],
                    }
                )

            result, contribution = await guarded_dispatch(
                "coherence_validator",
                "validate_estimate",
                {
                    "components": validate_rows,
                    "total_hours": float(estimate.get("total_hours") or 0.0),
                },
                step=step,
                estimation_id=estimation_id,
                privilege_strict=deps.privilege_strict,
                audit_preview_chars=deps.audit_preview_chars,
            )

            validation = {
                "ok": bool(result.get("ok", False)),
                "issues": list(result.get("issues") or []),
                "summary": result.get("summary"),
            }
            # Unbudgeted components already reduce coverage; counting every
            # "no historical reference" issue again would collapse confidence to
            # ~0 on any partially grounded estimate and force HITL forever.
            hard_issues = [
                issue
                for issue in validation["issues"]
                if "no historical reference" not in issue.lower()
                and "unbudgeted" not in issue.lower()
            ]
            provisional = {
                **state,
                "validation": {**validation, "issues": hard_issues},
            }
            confidence = compute_confidence(
                provisional,
                grounding_max_distance=deps.grounding_max_distance,
            )
            near = precedent_matches(matches, max_distance=deps.grounding_max_distance)
            matched_ids = {m.get("component_id") for m in near if m.get("component_id")}
            grounded = sum(
                1 for c in components if c.get("component_id") in matched_ids
            )
            out_of_range = is_outside_historical_band(
                estimate_for_historical_band(state, near),
                near,
                factor=deps.out_of_range_factor,
            )
            risk_flags = detect_review_risks(state)
            status = (
                "validated" if validation["ok"] and not out_of_range else "needs_review"
            )

            update: dict[str, Any] = {
                "validation": validation,
                "confidence": confidence,
                "out_of_range": out_of_range,
                "grounded_components": grounded,
                "risk_flags": risk_flags,
                "status": status,
                "agent_contributions": [contribution],
            }
            if validation["issues"]:
                update["errors"] = list(validation["issues"])
            return update

    async def human_review_gate(state: SupervisorState) -> dict[str, Any]:
        needs, reasons = requires_human_review(
            state,
            confidence_threshold=deps.confidence_threshold,
            out_of_range_factor=deps.out_of_range_factor,
            min_grounded_ratio=deps.min_grounded_ratio,
            grounding_max_distance=deps.grounding_max_distance,
        )
        if not needs:
            with logfire.span("gate: human_review (auto-approved)"):
                log.info(
                    "human_review_gate_skipped",
                    confidence=state.get("confidence"),
                    status=state.get("status"),
                )
                return {"needs_human_review": False, "review_reasons": []}

        # interrupt() FIRST — before any reducer write — because resume re-executes.
        decision = interrupt(
            {
                "gate": "low_confidence_review",
                "estimation_id": state.get("estimation_id"),
                "reasons": reasons,
                "confidence": state.get("confidence"),
                "threshold": deps.confidence_threshold,
                "estimate": state.get("estimate"),
                "validation": state.get("validation"),
                "risk_flags": state.get("risk_flags") or [],
                "routing_history": state.get("routing_history") or [],
            }
        )

        with logfire.span("gate: human_review"):
            decision = decision or {}
            applied = apply_human_decision(state, decision)
            action = decision.get("action") or decision.get("decision") or "approve"
            log.info(
                "human_review_gate_resumed",
                action=action,
                status=applied.get("status"),
                reasons=len(reasons),
            )
            return {
                **applied,
                "needs_human_review": True,
                "review_reasons": reasons,
                "agent_contributions": [
                    {
                        "step": _step_of(state),
                        "agent": "human",
                        "action": "review_decision",
                        "tool": None,
                        "outcome": "ok",
                        "summary": f"human {action}: {decision.get('note') or '—'}",
                        "args_digest": None,
                        "duration_ms": None,
                    }
                ],
            }

    return {
        "supervisor": supervisor,
        "requirements_extractor": requirements_extractor,
        "budget_searcher": budget_searcher,
        "estimate_generator": estimate_generator,
        "coherence_validator": coherence_validator,
        "human_review_gate": human_review_gate,
    }
