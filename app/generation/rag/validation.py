"""Post-generation checks for grounded estimates (Session 9)."""

from __future__ import annotations

from app.generation.rag.schemas import Estimate, RetrievedChunk


def validate_citations(
    estimate: Estimate,
    retrieved_chunks: list[RetrievedChunk],
) -> list[int]:
    """Return the cited source ids that were never retrieved (fabricated)."""
    valid_ids = {chunk.id for chunk in retrieved_chunks}

    cited_ids: set[int] = {citation.source_id for citation in estimate.sources}
    for module in estimate.modules:
        for task in module.tasks:
            cited_ids.update(task.sources)

    return sorted(cited_ids - valid_ids)


def check_coherence(estimate: Estimate) -> bool:
    """Return whether the estimate's confidence level matches its content."""
    if estimate.confidence != "insufficient":
        return True
    return (
        estimate.total_engineer_days is None
        and estimate.duration_weeks is None
        and not estimate.modules
        and bool(estimate.insufficient_context_explanation)
    )
