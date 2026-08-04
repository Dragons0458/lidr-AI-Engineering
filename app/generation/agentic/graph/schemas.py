"""Node-internal LLM I/O models for the multi-agent graph.

These are the ``response_model``s structured-output nodes validate against via
Instructor. They are private plumbing of the graph — the public HTTP contract
lives in ``app/domain/schemas/graph_estimation.py``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

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


# --------------------------------------------------------------------------- #
# Session 14 LIVE — competition schemas (hours, not engineer-days)            #
# --------------------------------------------------------------------------- #

Stance = Literal["conservative", "aggressive"]


class EstimateProposal(BaseModel):
    """One competing estimate produced from a single stance."""

    stance: Stance = Field(description="Which estimator produced this proposal.")
    total_hours: float = Field(
        ge=0, description="This stance's headline effort in hours."
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="The load-bearing assumptions this number rests on.",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="What could make the real effort diverge from this number.",
    )
    reasoning: str = Field(description="One paragraph: how the number was reached.")


class SynthesizedEstimate(BaseModel):
    """The synthesizer's output: a RANGE in hours, never an average."""

    low: float = Field(ge=0, description="Lower bound of the estimate range (hours).")
    high: float = Field(ge=0, description="Upper bound of the estimate range (hours).")
    driving_assumptions: list[str] = Field(
        default_factory=list,
        description="The assumptions that most move the number between low and high.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Questions whose answers would narrow the range.",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        default="medium", description="Confidence in the range as a whole."
    )
    reasoning: str = Field(
        description="Short prose on how the bracket was set — explicitly NOT an average."
    )

    @model_validator(mode="after")
    def _order_bounds(self) -> SynthesizedEstimate:
        if self.high < self.low:
            import structlog

            structlog.get_logger().warning(
                "synthesized_estimate_bounds_swapped",
                low=self.low,
                high=self.high,
            )
            self.low, self.high = self.high, self.low
        return self
