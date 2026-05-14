from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class ReferenceProject(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    summary: str = Field(..., min_length=10, max_length=600)
    estimated_hours: int = Field(..., ge=1, le=20000)
    team: str = Field(..., min_length=2, max_length=120)
    outcome: str = Field(..., min_length=5, max_length=300)


class EstimationRequest(BaseModel):
    transcript: str = Field(
        ..., description="Transcript of the meeting summary", min_length=10, max_length=2000
    )
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat
    reference_projects: list[ReferenceProject] | None = None


class TokenUsage(BaseModel):
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None


class EstimationResponse(BaseModel):
    estimation: str
    model: str
    provider: str
    timestamp: datetime
    usage: TokenUsage
    prompt_version: str
