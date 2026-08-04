"""Competition pattern (Session 14 LIVE): two estimators disagree, on purpose.

Fan-out / fan-in subgraph::

              ┌──▶ conservative_estimator ──┐
    START ────┤                             ├──▶ synthesizer ──▶ END
              └──▶ aggressive_estimator ────┘

``compute_divergence`` is pure arithmetic. The synthesizer is forbidden from averaging.
The subgraph is compiled per ``SupervisorDeps`` (injectable propose/synthesize callables),
never as a module-level singleton.
"""

from __future__ import annotations

import operator
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any, Optional

import structlog
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.generation.agentic.graph.schemas import EstimateProposal, SynthesizedEstimate

if TYPE_CHECKING:
    from app.generation.agentic.graph.supervisor_nodes import SupervisorDeps

log = structlog.get_logger()

ProposeFn = Callable[[str, str], Awaitable[EstimateProposal]]
SynthesizeFn = Callable[[list[dict], dict], Awaitable[SynthesizedEstimate]]


class CompetitionState(TypedDict, total=False):
    """Private subgraph state. ``proposals`` fan-in via ``operator.add``."""

    brief: str
    proposals: Annotated[list[dict], operator.add]
    divergence: Optional[dict]
    synthesis: Optional[dict]


_CONSERVATIVE_SYSTEM_PROMPT = (
    "You are a RISK-FIRST senior estimator. You have seen projects blow past their "
    "estimate and you price that memory in. Read the components, their scope and the "
    "historical references, then produce a single total in HOURS under these "
    "criteria:\n"
    "- Weight INTEGRATION FRICTION heavily: undocumented or legacy interfaces, "
    "third-party SDKs with thin docs, protocols nobody owns anymore.\n"
    "- Treat VAGUE or open scope as LARGER, not smaller — unresolved scope becomes work.\n"
    "- Price CERTIFICATION / COMPLIANCE overhead (SOC2, FIPS, audits) as real effort.\n"
    "- Count TECHNICAL UNKNOWNS (first-of-its-kind work, no internal precedent) as a "
    "multiplier, not a footnote.\n"
    "Prefer the CONSERVATIVE EVIDENCE section of the brief (highest historical hours "
    "first; components without precedent are marked).\n"
    "Return your total, the assumptions it rests on, the risks that justify the caution, "
    "and one paragraph of reasoning. Set stance to 'conservative'."
)

_AGGRESSIVE_SYSTEM_PROMPT = (
    "You are a REUSE-FIRST senior estimator. You have seen teams gold-plate estimates "
    "out of fear and lose the bid. Read the components, their scope and the historical "
    "references, then produce a single total in HOURS under these criteria:\n"
    "- Weight REUSE heavily: strong historical analogs, standard patterns, libraries and "
    "internal components the team already ships.\n"
    "- Treat CLOSED, well-understood scope as containable — do not pad for hypotheticals.\n"
    "- Trust the HISTORICAL REFERENCE HOURS as the base case when analogs are close.\n"
    "- Assume a COMPETENT team executing familiar work at a normal pace.\n"
    "Prefer the AGGRESSIVE EVIDENCE section of the brief (closest distance first; strong "
    "analogs are marked).\n"
    "Return your total, the assumptions it rests on, the risks you are consciously "
    "accepting, and one paragraph of reasoning. Set stance to 'aggressive'."
)

_SYNTHESIZER_SYSTEM_PROMPT = (
    "You are the estimation LEAD reconciling two independent estimates of the same "
    "project: a risk-first (conservative) and a reuse-first (aggressive) one. You are "
    "given both proposals and the ARITHMETIC divergence between their totals.\n"
    "\n"
    "DO NOT AVERAGE the two numbers. An average destroys the only useful thing the "
    "disagreement produced. Instead:\n"
    "- Return a RANGE [low, high] in HOURS that brackets the honest uncertainty. "
    "Anchor low near the reuse-first total and high near the risk-first total; widen "
    "it if the divergence is large.\n"
    "- List the DRIVING ASSUMPTIONS: the few beliefs that most move the number between "
    "low and high.\n"
    "- List the OPEN QUESTIONS whose answers would let a human collapse the range.\n"
    "- Set confidence to reflect the SPREAD: a wide divergence is 'low' confidence in a "
    "single number, however sure each estimator was of their own.\n"
    "Write one paragraph of reasoning that explains the bracket, not an average."
)


def _proposal_lines(proposals: list[dict]) -> str:
    lines: list[str] = []
    for proposal in proposals:
        lines.append(
            f"- [{proposal.get('stance')}] total = {proposal.get('total_hours')} hours\n"
            f"    assumptions: {'; '.join(proposal.get('assumptions') or []) or '—'}\n"
            f"    risks: {'; '.join(proposal.get('risks') or []) or '—'}"
        )
    return "\n".join(lines)


def compute_divergence(proposals: list[dict]) -> dict:
    """How far apart the two estimates are — pure arithmetic, no model judgement."""
    totals = [
        float(p["total_hours"]) for p in proposals if p.get("total_hours") is not None
    ]
    if len(totals) < 2:
        only = totals[0] if totals else 0.0
        return {
            "low": only,
            "high": only,
            "spread": 0.0,
            "ratio": 0.0,
            "level": "low",
        }

    low, high = min(totals), max(totals)
    spread = high - low
    midpoint = (low + high) / 2
    ratio = (spread / midpoint) if midpoint else 0.0
    level = "high" if ratio >= 0.5 else "medium" if ratio >= 0.2 else "low"
    return {
        "low": low,
        "high": high,
        "spread": spread,
        "ratio": round(ratio, 3),
        "level": level,
    }


def _component_refs(
    component: dict[str, Any], matches: list[dict]
) -> list[dict[str, Any]]:
    """Historical matches for one component (by component_id or name)."""
    cid = str(component.get("component_id") or "")
    name = str(component.get("name") or "")
    refs: list[dict[str, Any]] = []
    for match in matches:
        mid = str(match.get("component_id") or "")
        if mid and mid == cid:
            refs.append(match)
            continue
        if name and name.lower() in str(match.get("content") or "").lower():
            refs.append(match)
    return refs


def build_competition_brief(
    state: dict[str, Any], base_estimate: dict[str, Any]
) -> str:
    """Shared evidence with stance-biased orderings (highest impact on divergence).

    Conservative listing: references sorted by hours descending; unpreceded marked.
    Aggressive listing: references sorted by distance ascending; strong analogs marked.
    """
    components = state.get("components") or []
    matches = state.get("budget_matches") or []
    lines = [
        "Project components and historical reference budgets (hours).",
        f"Grounded consolidation total: {base_estimate.get('total_hours')} hours.",
        "",
        "## CONSERVATIVE EVIDENCE (highest historical hours first; no-precedent marked)",
    ]
    for component in components:
        refs = _component_refs(component, matches)
        if not refs:
            lines.append(
                f"- {component.get('name')} [{component.get('category')}]: "
                "NO PRECEDENT — treat as open risk"
            )
            continue
        ordered = sorted(
            refs,
            key=lambda m: float(m.get("amount") or m.get("estimated_hours") or 0.0),
            reverse=True,
        )
        ref_text = ", ".join(
            f"{float(m.get('amount') or m.get('estimated_hours') or 0.0):.0f}h"
            for m in ordered
        )
        lines.append(
            f"- {component.get('name')} [{component.get('category')}]: {ref_text}"
        )

    lines.append("")
    lines.append(
        "## AGGRESSIVE EVIDENCE (closest distance first; strong analogs marked)"
    )
    for component in components:
        refs = _component_refs(component, matches)
        if not refs:
            lines.append(
                f"- {component.get('name')} [{component.get('category')}]: "
                "no historical references"
            )
            continue
        ordered = sorted(
            refs,
            key=lambda m: float(
                m["distance"] if m.get("distance") is not None else 99.0
            ),
        )
        parts: list[str] = []
        for match in ordered:
            hours = float(match.get("amount") or match.get("estimated_hours") or 0.0)
            distance = match.get("distance")
            strong = distance is not None and float(distance) <= 0.25
            tag = " STRONG ANALOG" if strong else ""
            dist_txt = f" d={float(distance):.2f}" if distance is not None else ""
            parts.append(f"{hours:.0f}h{dist_txt}{tag}")
        lines.append(
            f"- {component.get('name')} [{component.get('category')}]: "
            + ", ".join(parts)
        )
    return "\n".join(lines)


def build_competition_subgraph(deps: SupervisorDeps):  # noqa: F821 — TYPE_CHECKING
    """Compile the fan-out / fan-in competition subgraph closed over ``deps``."""

    async def conservative_estimator(state: CompetitionState) -> dict:
        if deps.propose_estimate is None:
            raise RuntimeError("propose_estimate is required for competition")
        proposal = await deps.propose_estimate("conservative", state["brief"])
        data = (
            proposal.model_dump() if hasattr(proposal, "model_dump") else dict(proposal)
        )
        data["stance"] = "conservative"
        log.info("competition_conservative", total=data.get("total_hours"))
        return {"proposals": [data]}

    async def aggressive_estimator(state: CompetitionState) -> dict:
        if deps.propose_estimate is None:
            raise RuntimeError("propose_estimate is required for competition")
        proposal = await deps.propose_estimate("aggressive", state["brief"])
        data = (
            proposal.model_dump() if hasattr(proposal, "model_dump") else dict(proposal)
        )
        data["stance"] = "aggressive"
        log.info("competition_aggressive", total=data.get("total_hours"))
        return {"proposals": [data]}

    async def synthesizer(state: CompetitionState) -> dict:
        if deps.synthesize is None:
            raise RuntimeError("synthesize is required for competition")
        proposals = state.get("proposals") or []
        divergence = compute_divergence(proposals)
        synthesis = await deps.synthesize(proposals, divergence)
        data = (
            synthesis.model_dump()
            if hasattr(synthesis, "model_dump")
            else dict(synthesis)
        )
        log.info(
            "competition_synthesis",
            low=data.get("low"),
            high=data.get("high"),
            divergence=divergence["ratio"],
        )
        return {"divergence": divergence, "synthesis": data}

    builder = StateGraph(CompetitionState)
    builder.add_node("conservative_estimator", conservative_estimator)
    builder.add_node("aggressive_estimator", aggressive_estimator)
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "conservative_estimator")
    builder.add_edge(START, "aggressive_estimator")
    builder.add_edge("conservative_estimator", "synthesizer")
    builder.add_edge("aggressive_estimator", "synthesizer")
    builder.add_edge("synthesizer", END)
    return builder.compile()
