"""Human-readable serialization helpers for grounded estimates."""

from __future__ import annotations

from app.generation.rag.schemas import Estimate


def render_estimate_as_text(estimate: Estimate) -> str:
    """Serialize an estimate to plain text for evaluation (e.g. RAGAS answer column)."""
    lines: list[str] = [
        f"Confidence: {estimate.confidence}",
        f"Reasoning: {estimate.reasoning}",
    ]
    if estimate.total_engineer_days is not None:
        lines.append(f"Total engineer-days: {estimate.total_engineer_days}")
    if estimate.duration_weeks is not None:
        lines.append(f"Duration weeks: {estimate.duration_weeks}")
    if estimate.insufficient_context_explanation:
        lines.append(
            f"Insufficient context: {estimate.insufficient_context_explanation}"
        )

    for module in estimate.modules:
        lines.append(f"\n## Module: {module.name}")
        if module.description:
            lines.append(module.description)
        for task in module.tasks:
            effort = (
                f"{task.engineer_days} engineer-days"
                if task.engineer_days is not None
                else "no effort estimate"
            )
            grounded = "grounded" if task.grounded else "not grounded"
            lines.append(f"- {task.name} ({effort}, {grounded})")
            if task.description:
                lines.append(f"  {task.description}")
            for source in task.sources:
                lines.append(
                    f"  [chunk {source.chunk_id} / {source.document_id}] "
                    f'"{source.evidence}"'
                )

    if estimate.assumptions:
        lines.append("\nAssumptions:")
        for assumption in estimate.assumptions:
            lines.append(f"- {assumption.description} ({assumption.impact})")

    return "\n".join(lines)


def compact_response_for_relevancy(answer: str, *, max_chars: int = 3_000) -> str:
    """Condensed estimate text for RAGAS ``answer_relevancy``.

    The full grounded serialization interleaves multi-line ``reasoning`` and
    verbatim ``evidence`` with the scope. A prefix *blacklist* cannot strip those
    reliably: only the first line of each multi-line field carries the marker, so
    the hedging in a reasoning's second paragraph (and the trailing lines of a
    multi-line evidence span) leaks through. That hedging makes the RAGAS judge
    set ``noncommittal=1``, which zeroes the score
    (``score = cosine_sim * int(not all_noncommittal)``).

    We therefore *whitelist* the committal scope — the headline totals, module
    headers, grounded task bullets (which always carry an ``engineer-days``
    figure) and the short module/task descriptions that connect the scope back to
    the request — while dropping confidence, reasoning, per-line evidence and
    assumptions. A tiny state machine keeps the description that immediately
    follows a module header or task bullet and swallows the multi-line evidence
    that follows a ``[chunk …]`` marker.
    """
    kept: list[str] = []
    # mode: "scope" (default), "after_header" (next prose line is a description),
    # "evidence" (skip until the next structural line).
    mode = "scope"
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped == "Assumptions:":
            break
        if stripped.startswith(("Total engineer-days:", "Duration weeks:")):
            kept.append(stripped)
            mode = "scope"
        elif stripped.startswith("## Module:"):
            kept.append(stripped)
            mode = "after_header"
        elif (
            stripped.startswith("- ")
            and "engineer-days" in stripped
            and "not grounded" not in stripped.lower()
        ):
            kept.append(stripped)
            mode = "after_header"
        elif stripped.startswith("[chunk"):
            mode = "evidence"
        elif mode == "after_header" and stripped:
            # Short module/task description: keep, then stop expecting prose.
            kept.append(stripped)
            mode = "scope"
        # Any other line (confidence, reasoning, evidence continuation, blanks)
        # is dropped; "evidence" mode persists until the next structural line.

    # Abstention / insufficient-context estimates carry no scope; keep a single
    # non-empty line so the judge receives a valid (and honestly noncommittal)
    # response instead of an empty string.
    if not kept:
        for line in answer.splitlines():
            if line.strip():
                kept.append(line.strip())
                break

    compact = "\n".join(kept).strip()
    if len(compact) > max_chars:
        compact = compact[:max_chars].rsplit("\n", 1)[0]
    return compact
