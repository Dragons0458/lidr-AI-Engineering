"""Post-generation checks for grounded estimates (Session 9 / 11)."""

from __future__ import annotations

import structlog

from app.generation.rag.schemas import Estimate, RetrievedChunk, TaskItem

log = structlog.get_logger()


class LineRef:
    """Identifies one task line for citation reporting."""

    __slots__ = ("module_name", "task_name", "cited_chunk_ids")

    def __init__(
        self,
        *,
        module_name: str,
        task_name: str,
        cited_chunk_ids: list[int],
    ) -> None:
        self.module_name = module_name
        self.task_name = task_name
        self.cited_chunk_ids = cited_chunk_ids

    def as_dict(self) -> dict:
        return {
            "module": self.module_name,
            "task": self.task_name,
            "cited_chunk_ids": self.cited_chunk_ids,
        }


class CitationReport:
    """Outcome of line-level citation verification."""

    __slots__ = ("grounded", "dangling", "insufficient")

    def __init__(
        self,
        *,
        grounded: list[LineRef],
        dangling: list[LineRef],
        insufficient: list[LineRef],
    ) -> None:
        self.grounded = grounded
        self.dangling = dangling
        self.insufficient = insufficient

    @property
    def has_dangling(self) -> bool:
        return bool(self.dangling)


def _line_ref(module_name: str, task: TaskItem) -> LineRef:
    return LineRef(
        module_name=module_name,
        task_name=task.name,
        cited_chunk_ids=[ref.chunk_id for ref in task.sources],
    )


def verify_citations(
    estimate: Estimate,
    retrieved_chunks: list[RetrievedChunk],
) -> CitationReport:
    """Classify every task line by citation integrity against retrieved context."""
    valid_ids = {chunk.id for chunk in retrieved_chunks}
    grounded: list[LineRef] = []
    dangling: list[LineRef] = []
    insufficient: list[LineRef] = []

    for module in estimate.modules:
        for task in module.tasks:
            ref = _line_ref(module.name, task)
            if not task.grounded:
                insufficient.append(ref)
                continue
            cited = {source.chunk_id for source in task.sources}
            if cited - valid_ids:
                dangling.append(ref)
            else:
                grounded.append(ref)

    return CitationReport(
        grounded=grounded, dangling=dangling, insufficient=insufficient
    )


def log_citation_report(report: CitationReport, *, request_id: str) -> None:
    """Emit structured logs for the citation verification outcome."""
    log.info(
        "citation_verification",
        request_id=request_id,
        grounded=len(report.grounded),
        dangling=len(report.dangling),
        insufficient=len(report.insufficient),
    )
    if report.dangling:
        log.warning(
            "dangling_citations",
            request_id=request_id,
            lines=[line.as_dict() for line in report.dangling],
        )


def degrade_dangling_tasks(estimate: Estimate, report: CitationReport) -> Estimate:
    """Downgrade tasks with dangling citations to ungrounded (no invented attribution)."""
    dangling_keys = {(line.module_name, line.task_name) for line in report.dangling}
    if not dangling_keys:
        return estimate

    updated_modules = []
    for module in estimate.modules:
        updated_tasks = []
        for task in module.tasks:
            if (module.name, task.name) in dangling_keys:
                updated_tasks.append(
                    task.model_copy(
                        update={"grounded": False, "sources": [], "engineer_days": None}
                    )
                )
            else:
                updated_tasks.append(task)
        updated_modules.append(module.model_copy(update={"tasks": updated_tasks}))
    return estimate.model_copy(update={"modules": updated_modules, "confidence": "low"})


def validate_citations(
    estimate: Estimate,
    retrieved_chunks: list[RetrievedChunk],
) -> list[int]:
    """Return cited chunk ids that were never retrieved (fabricated).

    Backward-compatible flat view used by stage endpoints; prefer
    :func:`verify_citations` for the full line-level report.
    """
    report = verify_citations(estimate, retrieved_chunks)
    dangling_ids: set[int] = set()
    for line in report.dangling:
        dangling_ids.update(line.cited_chunk_ids)

    valid_ids = {chunk.id for chunk in retrieved_chunks}
    cited_ids: set[int] = {citation.source_id for citation in estimate.sources}
    for module in estimate.modules:
        for task in module.tasks:
            cited_ids.update(ref.chunk_id for ref in task.sources)

    return sorted((cited_ids - valid_ids) | dangling_ids)


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
