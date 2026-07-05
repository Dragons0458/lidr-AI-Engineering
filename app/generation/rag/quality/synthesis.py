"""Contradiction synthesis for per-task hours (Session 11).

When historical neighbours disagree beyond a dispersion threshold, expose an
``HourRange`` alongside the consensus point estimate.
"""

from __future__ import annotations

import asyncio

import structlog
from pydantic import BaseModel, Field

from app.generation.rag.schemas import HourRange, TaskNeighbor

log = structlog.get_logger()


def is_contradiction(dispersion: float | None, threshold: float) -> bool:
    """Return True when neighbour dispersion crosses the contradiction threshold."""
    if dispersion is None:
        return False
    return dispersion >= threshold


def _deterministic_range(neighbors: list[TaskNeighbor]) -> HourRange:
    hours = [n.estimated_hours for n in neighbors]
    low, high = min(hours), max(hours)
    names = ", ".join(sorted({n.budget_id or str(n.source_id) for n in neighbors}))
    return HourRange(
        low=low,
        high=high,
        reason=(
            f"Historical neighbours disagree on hours ({low}–{high}h); "
            f"conflicting sources: {names}."
        ),
    )


class _Reason(BaseModel):
    reason: str = Field(description="One-sentence explanation of the hour spread.")


async def synthesize_range(
    neighbors: list[TaskNeighbor],
    dispersion: float | None,
    *,
    threshold: float,
    use_llm: bool = False,
    model: str | None = None,
) -> HourRange | None:
    """Return an hour range when neighbours contradict; ``None`` otherwise."""
    if not is_contradiction(dispersion, threshold) or not neighbors:
        return None

    base = _deterministic_range(neighbors)
    if not use_llm:
        return base

    from app.dependencies import get_llm_wrapper

    lines = "\n".join(
        f"- {n.budget_id or n.source_id}: {n.estimated_hours}h (d={n.distance:.3f})"
        for n in neighbors
    )
    try:
        wrapper = get_llm_wrapper()
        panel, _meta = await asyncio.to_thread(
            wrapper.complete_structured,
            system_prompt=(
                "You explain why historical task-hour neighbours disagree. "
                "Be concise and name the conflict."
            ),
            user_message=(
                f"Neighbours:\n{lines}\n\n"
                f"Consensus spread: {base.low}–{base.high} hours. "
                "Return a one-sentence reason."
            ),
            response_model=_Reason,
            model_override=model,
        )
        return HourRange(low=base.low, high=base.high, reason=panel.reason)
    except Exception as exc:  # noqa: BLE001
        log.warning("synthesis_reason_failed", error=str(exc)[:200])
        return base
