from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class EstimationRequest(BaseModel):
    transcript: str = Field(
        ..., description="Transcript of the meeting summary", min_length=10
    )


class TokenUsage(BaseModel):
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None


class EstimationResponse(BaseModel):
    estimation: str
    model: str
    provider: str
    timestamp: datetime
    usage: TokenUsage
