"""HTTP contracts for the Session 13 LangGraph estimation endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.generation.agentic.agent_schemas import AgentEstimate

GraphStatus = Literal["validated", "needs_review"]


class GraphEstimationRequest(BaseModel):
    """Client-owned estimation identity plus the meeting transcript."""

    model_config = ConfigDict(extra="forbid")

    estimation_id: str = Field(min_length=1, max_length=200)
    transcript: str = Field(min_length=1)


class GraphEstimationResponse(BaseModel):
    """Public graph result: structured estimate plus consolidation status."""

    estimate: AgentEstimate
    status: GraphStatus
