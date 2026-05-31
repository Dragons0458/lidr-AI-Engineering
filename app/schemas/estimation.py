from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.attachments import AttachmentText

PreprocessingMode = Literal["none", "inline_cleaning", "two_phase"]
ExampleFormat = Literal["markdown", "json", "narrative"]


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
    description: str = Field(
        ...,
        description="Project description, meeting summary, or latest session input",
        min_length=10,
        max_length=12000,
    )
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat
    reference_projects: list[ReferenceProject] | None = None
    attachments: list[AttachmentText] | None = None
    preprocessing: PreprocessingMode = "none"
    use_examples: bool = True
    num_examples: int = Field(default=3, ge=0, le=5)
    example_format: ExampleFormat = "markdown"
    model: str | None = None
    max_tokens: int = Field(default=4000, ge=256, le=16000)
    thinking_budget: int | None = Field(default=None, ge=0, le=16000)
    evaluate: bool = True


class TokenUsage(BaseModel):
    tokens_used: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    preprocessing_input_tokens: int = 0
    preprocessing_output_tokens: int = 0
    cost_estimate: Optional[float] = None


class StructureCheck(BaseModel):
    has_title: bool
    has_breakdown_table: bool
    has_totals_section: bool
    has_team_section: bool
    has_duration_section: bool
    declared_total_hours: int | None = None
    sum_row_hours: int | None = None
    hours_match: bool | None = None
    declared_total_cost: float | None = None
    sum_row_cost: float | None = None
    cost_match: bool | None = None
    finish_reason_ok: bool
    score: float
    issues: list[str] = Field(default_factory=list)


class EstimationResponse(BaseModel):
    estimation: str
    model: str
    provider: str
    timestamp: datetime
    usage: TokenUsage
    prompt_version: str
    project_metadata: dict[str, object] | None = None
    latency_ms: int = 0
    finish_reason: str | None = None
    preprocessing: PreprocessingMode = "none"
    extracted_requirements: str | None = None
    validation: StructureCheck | None = None
    cache_hit: bool = False
    cost_usd: float = 0.0
    out_of_scope: bool = False
