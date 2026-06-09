from typing import Literal

from pydantic import BaseModel, Field

from app.domain.schemas.estimation import StructuredEstimationResponse

BossDecision = Literal["accept", "iterate", "synthesize"]


class ACBIteration(BaseModel):
    iteration: int
    decision_after: BossDecision
    critic_verdict: str
    critic_confidence: int
    issue_summary: list[str] = Field(default_factory=list)


class BossTrace(BaseModel):
    iterations: list[ACBIteration] = Field(default_factory=list)
    final_decision: BossDecision
    iterations_run: int


class ACBResponse(StructuredEstimationResponse):
    acb: BossTrace
