"""Semantic hallucination gate for grounded estimates (Session 11).

Combines a numeric anchor (cited chunk hours) with an optional LLM judge to
flag lines that over-estimate or are not entailed by the retrieved evidence.
"""

from __future__ import annotations

import asyncio
import re

import structlog
from pydantic import BaseModel, Field

from app.generation.rag.schemas import (
    Estimate,
    HallucinationReport,
    LineGate,
    LineVerdict,
    RetrievedChunk,
    TaskItem,
)

log = structlog.get_logger()

_HOURS_PER_DAY = 8
_HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*h(?:ours?)?", re.IGNORECASE)


def _chunk_hours(chunk: RetrievedChunk) -> float | None:
    if chunk.estimated_hours is not None:
        return float(chunk.estimated_hours)
    match = _HOURS_RE.search(chunk.content)
    return float(match.group(1)) if match else None


def numeric_anchor(
    task: TaskItem,
    chunks_by_id: dict[int, RetrievedChunk],
) -> float | None:
    """Sum cited chunk hours and convert to engineer-days."""
    total_hours = 0.0
    found = False
    for source in task.sources:
        chunk = chunks_by_id.get(source.chunk_id)
        if chunk is None:
            continue
        hours = _chunk_hours(chunk)
        if hours is not None:
            total_hours += hours
            found = True
    if not found:
        return None
    return total_hours / _HOURS_PER_DAY


def gate_line(
    task: TaskItem,
    module_name: str,
    anchor_days: float | None,
    verdict: LineVerdict | None,
    *,
    tolerance: float,
) -> LineGate:
    """Pure per-line gate: asymmetric numeric overage + optional judge verdict."""
    if not task.grounded:
        return LineGate(
            module=module_name,
            component=task.name,
            status="insufficient",
            reason="task not grounded",
        )

    numeric_fail = False
    numeric_deviation: float | None = None
    reasons: list[str] = []

    if anchor_days is not None and anchor_days > 0 and task.engineer_days is not None:
        numeric_deviation = abs(task.engineer_days - anchor_days) / anchor_days
        overage = (task.engineer_days - anchor_days) / anchor_days
        if overage > tolerance:
            numeric_fail = True
            reasons.append(
                f"engineer_days {task.engineer_days} exceeds anchor "
                f"{anchor_days:.1f}d by {overage:.0%}"
            )

    judge_fail = verdict is not None and not verdict.entailed
    if judge_fail and verdict is not None:
        reasons.append(verdict.reason or "judge rejected entailed claim")

    if numeric_fail or judge_fail:
        return LineGate(
            module=module_name,
            component=task.name,
            status="degraded",
            numeric_deviation=numeric_deviation,
            reason="; ".join(reasons),
        )

    return LineGate(
        module=module_name,
        component=task.name,
        status="grounded",
        numeric_deviation=numeric_deviation,
        reason="",
    )


class _Panel(BaseModel):
    verdicts: list[LineVerdict] = Field(default_factory=list)


def _format_estimate_for_judge(estimate: Estimate) -> str:
    lines: list[str] = []
    for module in estimate.modules:
        for task in module.tasks:
            days = task.engineer_days if task.engineer_days is not None else "?"
            lines.append(f"{module.name} :: {task.name} :: {days} engineer-days")
    return "\n".join(lines)


async def judge_estimate(
    estimate: Estimate,
    *,
    model: str,
) -> dict[tuple[str, str], LineVerdict]:
    """One batched LLM call; default to not entailed on failure."""
    from app.dependencies import get_llm_wrapper

    user_message = _format_estimate_for_judge(estimate)
    if not user_message.strip():
        return {}

    try:
        wrapper = get_llm_wrapper()
        panel, _meta = await asyncio.to_thread(
            wrapper.complete_structured,
            system_prompt=(
                "You are a strict entailment judge. For each task line, decide whether "
                "the engineer-day claim is fully supported by typical historical budget "
                "evidence. Default to entailed=false when uncertain."
            ),
            user_message=user_message,
            response_model=_Panel,
            model_override=model,
        )
        return {(v.module, v.component): v for v in panel.verdicts}
    except Exception as exc:  # noqa: BLE001
        log.warning("hallucination_judge_failed", error=str(exc)[:200])
        return {}


async def gate_estimate(
    estimate: Estimate,
    chunks: list[RetrievedChunk],
    *,
    tolerance: float,
    judge_model: str,
    use_judge: bool = True,
) -> HallucinationReport:
    """Anchor every grounded line and optionally run the batched judge."""
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    verdicts = await judge_estimate(estimate, model=judge_model) if use_judge else {}

    lines: list[LineGate] = []
    grounded_lines = degraded_lines = insufficient_lines = 0

    for module in estimate.modules:
        for task in module.tasks:
            anchor = numeric_anchor(task, chunks_by_id) if task.grounded else None
            verdict = verdicts.get((module.name, task.name))
            gate = gate_line(
                task,
                module.name,
                anchor,
                verdict,
                tolerance=tolerance,
            )
            lines.append(gate)
            if gate.status == "grounded":
                grounded_lines += 1
            elif gate.status == "degraded":
                degraded_lines += 1
            else:
                insufficient_lines += 1

    return HallucinationReport(
        total_lines=len(lines),
        grounded_lines=grounded_lines,
        degraded_lines=degraded_lines,
        insufficient_lines=insufficient_lines,
        lines=lines,
    )
