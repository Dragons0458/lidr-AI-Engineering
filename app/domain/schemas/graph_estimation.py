"""HTTP contracts for the Session 13 LangGraph multi-agent estimation flow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.generation.agentic.agent_schemas import AgentEstimate

GraphStatus = Literal["validated", "needs_review"]


class GraphEstimationRequest(BaseModel):
    """Client-owned estimation identity plus the meeting transcript (START)."""

    model_config = ConfigDict(extra="forbid")

    estimation_id: str = Field(min_length=1, max_length=200)
    transcript: str = Field(min_length=1)


class GraphEstimationResponse(BaseModel):
    """Legacy sequential graph response — kept for older tests only."""

    estimate: AgentEstimate
    status: GraphStatus


class GraphResumeRequest(BaseModel):
    """Human decision payload for ``POST …/graph/{estimation_id}/resume``."""

    model_config = ConfigDict(extra="forbid")

    decision: dict = Field(default_factory=dict)


class PendingGate(BaseModel):
    """The human gate a paused run is waiting on (the ``interrupt`` payload)."""

    model_config = ConfigDict(extra="forbid")

    gate: str
    estimation_id: str
    payload: dict = Field(default_factory=dict)


class GraphRunState(BaseModel):
    """Snapshot of a multi-agent run: paused at a gate or completed."""

    model_config = ConfigDict(extra="forbid")

    estimation_id: str
    state: Literal["paused", "completed"]
    pending_gate: PendingGate | None = None
    complexity: str | None = None
    structure: dict | None = None
    task_hours: list[dict] = Field(default_factory=list)
    estimate: dict | None = None
    analysis_report: dict | None = None
    proposal: str | None = None
    status: str | None = None
    errors: list[str] = Field(default_factory=list)


class ActivityEntry(BaseModel):
    """One didactic line of what an agent just did."""

    model_config = ConfigDict(extra="forbid")

    seq: int = 0
    node: str
    label: str
    message: str
    ts: str | None = None


class GraphProgress(GraphRunState):
    """Live progress: ``GraphRunState`` plus the activity feed."""

    state: Literal["running", "paused", "completed"]  # type: ignore[assignment]
    activity: list[ActivityEntry] = Field(default_factory=list)


class GraphProposalResponse(BaseModel):
    """Commercial proposal drafted on demand from a validated estimate."""

    model_config = ConfigDict(extra="forbid")

    estimation_id: str
    title: str
    executive_summary: str
    scope: list[str] = Field(default_factory=list)
    total_engineer_days: int | None = None
    body_markdown: str
