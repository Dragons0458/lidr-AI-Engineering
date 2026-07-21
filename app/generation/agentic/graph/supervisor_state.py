"""Session 14 supervisor state: S13 EstimationState extended with keyed trails.

``SupervisorState`` subclasses ``EstimationState`` so parent reducers
(``budget_matches``, ``errors``, ``task_hours``) come along. Two new keyed
accumulators — ``routing_history`` and ``agent_contributions`` — stay
idempotent under interrupt/resume re-execution.

Signal helpers (``requires_human_review``, ``historical_band``, …) are pure
functions of state so the human gate can re-run safely on resume.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

from app.generation.agentic.graph.state import EstimationState


class AgentContribution(TypedDict, total=False):
    """One auditable action: model call, tool call, or denied tool attempt."""

    step: int
    agent: str
    action: str
    tool: Optional[str]
    outcome: str
    summary: str
    args_digest: Optional[str]
    duration_ms: Optional[int]


class RoutingRecord(TypedDict, total=False):
    """One supervisor decision with provenance (``llm`` | ``fallback`` | ``limit``)."""

    step: int
    next_agent: str
    reason: str
    source: str
    decision_confidence: Optional[str]


def _keyed_append(
    existing: list[dict] | None,
    new: list[dict] | None,
    *,
    key: Callable[[dict], tuple],
) -> list[dict]:
    """Append-only accumulator that merges on repeated keys (resume-safe)."""
    merged: dict[tuple, dict] = {}
    for item in list(existing or []) + list(new or []):
        item_key = key(item)
        merged[item_key] = {**merged.get(item_key, {}), **item}
    return list(merged.values())


def _contribution_key(contribution: dict) -> tuple:
    return (
        contribution.get("step"),
        contribution.get("agent"),
        contribution.get("action"),
        contribution.get("args_digest"),
    )


def _routing_key(record: dict) -> tuple:
    return (record.get("step"),)


def append_contributions(
    existing: list[dict] | None, new: list[dict] | None
) -> list[dict]:
    """Reducer for ``agent_contributions`` — keyed by step/agent/action/digest."""
    return _keyed_append(existing, new, key=_contribution_key)


def append_routing(existing: list[dict] | None, new: list[dict] | None) -> list[dict]:
    """Reducer for ``routing_history`` — one decision per step."""
    return _keyed_append(existing, new, key=_routing_key)


class SupervisorState(EstimationState, total=False):
    """Shared state for the supervisor star topology."""

    next_agent: Optional[str]
    route_reason: Optional[str]
    # Last-write-wins; MUST NOT use a reducer (would break the step budget on resume).
    supervisor_steps: int
    routing_history: Annotated[list[RoutingRecord], append_routing]
    agent_contributions: Annotated[list[AgentContribution], append_contributions]

    component_anchors: list[dict]
    validation: Optional[dict]
    confidence: Optional[float]
    out_of_range: Optional[bool]
    grounded_components: Optional[int]
    search_completed: Optional[bool]
    risk_flags: list[str]

    needs_human_review: Optional[bool]
    review_reasons: list[str]
    human_decision: Optional[dict]


def privilege_violations(state: dict[str, Any]) -> list[dict]:
    """Derived read-model: every denied action in the audit trail."""
    return [
        contribution
        for contribution in (state.get("agent_contributions") or [])
        if contribution.get("outcome") == "denied"
    ]


# --------------------------------------------------------------------------- #
# Signal helpers (pure)                                                       #
# --------------------------------------------------------------------------- #


_REVIEW_RISK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "novel_cryptography",
        re.compile(r"\bqkd\b|quantum key|distribuci[oó]n cu[aá]ntica", re.I),
    ),
    (
        "legacy_or_undocumented_integration",
        re.compile(
            r"\bcobol\b|\bmainframe\b|undocument|no documentad|"
            r"formato propietario|fixed[- ]length|longitud fija",
            re.I,
        ),
    ),
    (
        "custom_security_hardware",
        re.compile(r"biom[eé]tric|\biris\b|\bhsm\b|fips\s*140", re.I),
    ),
    (
        "extreme_throughput",
        re.compile(
            r"\b\d[\d., ]{3,}\s*(?:events|eventos)\s*"
            r"(?:per second|por segundo)",
            re.I,
        ),
    ),
    ("mandatory_certification", re.compile(r"\bsoc\s*2\b", re.I)),
    (
        "open_scope",
        re.compile(
            r"scope.{0,35}(?:open|not closed)|"
            r"alcance.{0,35}(?:abierto|no.{0,12}cerrad)|"
            r"seg[uú]n avancemos.{0,60}(?:salen|salgan|crezca)",
            re.I,
        ),
    ),
)


def detect_review_risks(state: dict[str, Any]) -> list[str]:
    """Detect deterministic novelty/risk signals that distance alone cannot express."""
    parts = [str(state.get("transcript") or "")]
    parts.extend(str(value) for value in (state.get("project_brief") or {}).values())
    parts.extend(str(value) for value in (state.get("requirements") or []))
    for component in state.get("components") or []:
        parts.extend(
            str(component.get(field) or "")
            for field in ("category", "name", "description")
        )
    text = "\n".join(parts)
    return [name for name, pattern in _REVIEW_RISK_PATTERNS if pattern.search(text)]


def build_state_digest(
    state: dict[str, Any],
    *,
    grounding_max_distance: float = 0.45,
) -> str:
    """Compact factual digest for the router — never includes the transcript."""
    components = state.get("components") or []
    matches = state.get("budget_matches") or []
    estimate = state.get("estimate")
    validation = state.get("validation")
    near = precedent_matches(matches, max_distance=grounding_max_distance)
    grounded = len({m.get("component_id") for m in near if m.get("component_id")})
    risk_flags = state.get("risk_flags") or detect_review_risks(state)
    parts = [
        f"requirements={len(state.get('requirements') or [])}",
        f"components={len(components)}",
        f"budget_matches={len(matches)}",
        f"grounded={grounded}/{len(components)}",
        f"risk_flags={','.join(risk_flags) if risk_flags else 'none'}",
        f"estimate={'yes' if estimate else 'no'}",
        f"validation={'yes' if validation else 'no'}",
        f"confidence={state.get('confidence')}",
        f"search_completed={bool(state.get('search_completed'))}",
        f"supervisor_steps={int(state.get('supervisor_steps') or state.get('routing_steps') or 0)}",
        f"status={state.get('status')}",
    ]
    digest = " ".join(parts)
    return digest[:500]


def precedent_matches(
    matches: list[dict],
    *,
    max_distance: float = 0.45,
) -> list[dict]:
    """Matches close enough to count as historical precedent (not false neighbours)."""
    near: list[dict] = []
    for match in matches:
        distance = match.get("distance")
        if distance is None:
            near.append(match)
            continue
        if float(distance) <= max_distance:
            near.append(match)
    return near


def compute_confidence(
    state: dict[str, Any],
    *,
    grounding_max_distance: float = 0.45,
) -> float:
    """Coverage of near-matched components minus 0.1 per validation issue."""
    components = state.get("components") or []
    total = len(components)
    if total == 0:
        coverage = 0.0
    else:
        near = precedent_matches(
            state.get("budget_matches") or [],
            max_distance=grounding_max_distance,
        )
        matched_ids = {m.get("component_id") for m in near if m.get("component_id")}
        covered = sum(1 for c in components if c.get("component_id") in matched_ids)
        coverage = covered / total
    issues = (state.get("validation") or {}).get("issues") or []
    return max(0.0, min(1.0, coverage - 0.1 * len(issues)))


def historical_band(matches: list[dict], factor: float = 2.0) -> dict[str, float]:
    """Sum of per-component medians; band is ``[centre/factor, centre*factor]``."""
    by_component: dict[str, list[float]] = {}
    for match in matches:
        component_id = match.get("component_id")
        if not component_id:
            continue
        by_component.setdefault(str(component_id), []).append(float(match["amount"]))
    if not by_component:
        return {"centre": 0.0, "low": 0.0, "high": 0.0}
    centre = sum(statistics.median(amounts) for amounts in by_component.values())
    if factor <= 0:
        factor = 1.0
    return {
        "centre": float(centre),
        "low": float(centre / factor),
        "high": float(centre * factor),
    }


def is_outside_historical_band(
    estimate: dict[str, Any],
    matches: list[dict],
    factor: float = 2.0,
) -> bool:
    """Whether ``estimate['total_hours']`` falls outside the historical band."""
    total = float(estimate.get("total_hours") or 0.0)
    band = historical_band(matches, factor=factor)
    if band["centre"] == 0.0 and not matches:
        return False
    return not (band["low"] <= total <= band["high"])


def estimate_for_historical_band(
    state: dict[str, Any], matches: list[dict]
) -> dict[str, float]:
    """Project an estimate onto the same grounded components as ``matches``.

    Comparing a full-project total with a band built from only the grounded
    subset creates false outliers. ``component_anchors`` preserves the stable
    component ids needed to compare like with like. Older/minimal states fall
    back to the complete estimate for backwards compatibility.
    """
    anchors = state.get("component_anchors") or []
    grounded_ids = {
        str(match.get("component_id")) for match in matches if match.get("component_id")
    }
    if not anchors or not grounded_ids:
        return {
            "total_hours": float(
                (state.get("estimate") or {}).get("total_hours") or 0.0
            )
        }
    total = sum(
        float(anchor.get("estimated_hours") or 0.0)
        for anchor in anchors
        if str(anchor.get("component_id")) in grounded_ids
    )
    return {"total_hours": total}


def requires_human_review(
    state: dict[str, Any],
    confidence_threshold: float = 0.7,
    out_of_range_factor: float = 2.0,
    min_grounded_ratio: float = 0.5,
    grounding_max_distance: float = 0.45,
) -> tuple[bool, list[str]]:
    """Pure HITL trigger from grounding, range, validation, and scope risk."""
    reasons: list[str] = []

    confidence = state.get("confidence")
    if confidence is not None and confidence < confidence_threshold:
        reasons.append("low_confidence")

    components = state.get("components") or []
    near = precedent_matches(
        state.get("budget_matches") or [],
        max_distance=grounding_max_distance,
    )
    total = len(components)
    matched_ids = {m.get("component_id") for m in near if m.get("component_id")}
    grounded = sum(1 for c in components if c.get("component_id") in matched_ids)

    if not near:
        reasons.append("no_precedent")
    elif total > 0 and (grounded / total) < min_grounded_ratio:
        reasons.append("no_precedent")

    estimate = state.get("estimate")
    if estimate and is_outside_historical_band(
        estimate_for_historical_band(state, near),
        near,
        factor=out_of_range_factor,
    ):
        reasons.append("out_of_range")

    if state.get("risk_flags") or detect_review_risks(state):
        reasons.append("high_risk_scope")

    # Preserve stable order, drop accidental duplicates.
    ordered: list[str] = []
    for reason in reasons:
        if reason not in ordered:
            ordered.append(reason)
    return bool(ordered), ordered


def apply_human_decision(state: dict[str, Any], decision: dict[str, Any]) -> dict:
    """Fold approve / adjust / reject into a partial state update."""
    decision = decision or {}
    action = decision.get("action") or decision.get("decision") or "approve"
    estimate = dict(state.get("estimate") or {})

    if action == "reject":
        return {"estimate": estimate, "status": "rejected", "human_decision": decision}

    if action == "adjust":
        overrides = decision.get("estimate_overrides") or {}
        estimate = {**estimate, **overrides}
        components = estimate.get("components") or []
        if components:
            estimate["total_hours"] = round(
                sum(float(c.get("estimated_hours") or 0.0) for c in components),
                1,
            )
        return {
            "estimate": estimate,
            "status": "needs_review",
            "human_decision": decision,
        }

    # approve (default)
    return {"estimate": estimate, "status": "validated", "human_decision": decision}
