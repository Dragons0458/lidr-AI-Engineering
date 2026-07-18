"""Node-internal LLM I/O models for the multi-agent graph.

These are the ``response_model``s structured-output nodes validate against via
Instructor. They are private plumbing of the graph — the public HTTP contract
lives in ``app/domain/schemas/graph_estimation.py``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Complexity = Literal["low", "medium", "high"]


class ComplexityClassification(BaseModel):
    """Output of ``classifier_agent``: complexity + a reformulated brief."""

    complexity: Complexity = Field(
        description="How complex the estimation is: low, medium or high."
    )
    reformulated_transcript: str = Field(
        min_length=1,
        description="Clean, self-contained project brief in technical English.",
    )
    reasoning: str = Field(description="One line on why that complexity was assigned.")


class WeakPoint(BaseModel):
    """One weakness the analysis agent flags for the human's final review."""

    area: str = Field(description="Module/task or cross-cutting concern.")
    issue: str = Field(description="What is uncertain, ungrounded or contradictory.")
    severity: Literal["low", "medium", "high"] = "medium"


class ReliabilityReport(BaseModel):
    """Output of ``analysis_agent``: a data-reliability read of the estimate."""

    overall_confidence: Literal["low", "medium", "high"] = Field(
        description="Overall confidence in the estimate as a whole."
    )
    grounded_task_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of tasks that got hours from a historical match.",
    )
    weak_points: list[WeakPoint] = Field(
        default_factory=list,
        description="Specific soft spots the human should check or complete.",
    )
    summary: str = Field(
        description="A short prose read of the estimate's reliability."
    )


class CommercialProposal(BaseModel):
    """Output of ``proposal_agent``: a client-facing commercial proposal."""

    title: str = Field(description="Proposal title.")
    executive_summary: str = Field(
        description="2-4 sentences a client executive would read."
    )
    scope: list[str] = Field(
        default_factory=list, description="Bullet scope of modules/deliverables."
    )
    total_engineer_days: int | None = Field(
        default=None, ge=0, description="Headline effort from the validated estimate."
    )
    body_markdown: str = Field(
        description="Full proposal as Markdown, grounded in the validated estimate."
    )
