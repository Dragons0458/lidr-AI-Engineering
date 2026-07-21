"""HTTP contract for the Session 14 supervisor multi-agent flow."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SupervisorEstimateRequest(BaseModel):
    """START a supervisor run."""

    transcript: str = Field(min_length=100, max_length=50_000)
    estimation_id: str | None = Field(default=None, max_length=128)


class SupervisorResumeRequest(BaseModel):
    """Typed human answer to the review gate."""

    decision: Literal["approve", "adjust", "reject"]
    estimate_overrides: dict | None = Field(
        default=None,
        description=(
            "Fields the reviewer edited; merged over the estimate. "
            "Only meaningful for 'adjust'."
        ),
    )
    note: str | None = Field(default=None, max_length=2000)


class PendingHumanReview(BaseModel):
    """Payload the reviewer needs while the run is paused."""

    gate: str = "low_confidence_review"
    estimation_id: str
    reasons: list[str] = Field(default_factory=list)
    confidence: float | None = None
    threshold: float | None = None
    estimate: dict | None = None
    validation: dict | None = None
    risk_flags: list[str] = Field(default_factory=list)


class SupervisorRunState(BaseModel):
    """Paused for a human, or completed."""

    estimation_id: str
    state: Literal["paused", "completed"]
    status: str = Field(
        description=(
            "validated | needs_review | rejected | awaiting_human_review. "
            "awaiting_human_review is derived while paused; never stored."
        )
    )
    pending_review: PendingHumanReview | None = None

    estimate: dict | None = None
    confidence: float | None = None
    requirements: list[str] = Field(default_factory=list)
    # Plain dicts (not graph.state TypedDicts) — Pydantic on Python <3.12 rejects
    # typing.TypedDict; shapes match Component / BudgetMatch from graph.state.
    components: list[dict[str, Any]] = Field(default_factory=list)
    budget_matches: list[dict[str, Any]] = Field(default_factory=list)
    validation: dict | None = None
    risk_flags: list[str] = Field(default_factory=list)
    human_decision: dict | None = None

    routing_history: list[dict] = Field(default_factory=list)
    agent_contributions: list[dict] = Field(default_factory=list)
    privilege_violations: list[dict] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
