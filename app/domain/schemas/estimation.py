from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.domain.schemas.attachments import AttachmentText

OUT_OF_SCOPE_PREFIX = "Out of scope:"
LOW_CONFIDENCE_THRESHOLD = 30

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


class Phase(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    base_hours: int = Field(ge=0, le=100_000)
    buffer_hours: int = Field(ge=0, le=100_000)
    team: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=10, max_length=600)


class EstimationResult(BaseModel):
    summary: str = Field(min_length=10, max_length=1200)
    confidence_pct: int = Field(ge=0, le=100)
    phases: list[Phase] = Field(min_length=1, max_length=12)
    total_base_hours: int = Field(ge=0, le=200_000)
    total_buffer_hours: int = Field(ge=0, le=200_000)
    total_hours: int = Field(ge=1, le=400_000)
    total_cost_eur: int = Field(ge=0, le=2_000_000)

    @model_validator(mode="after")
    def validate_totals_and_scope(self) -> EstimationResult:
        phase_base = sum(phase.base_hours for phase in self.phases)
        phase_buffer = sum(phase.buffer_hours for phase in self.phases)
        if self.total_base_hours != phase_base:
            raise ValueError("total_base_hours must equal sum of phase base_hours")
        if self.total_buffer_hours != phase_buffer:
            raise ValueError("total_buffer_hours must equal sum of phase buffer_hours")
        if self.total_hours != self.total_base_hours + self.total_buffer_hours:
            raise ValueError(
                "total_hours must equal total_base_hours + total_buffer_hours"
            )
        if (
            self.confidence_pct < LOW_CONFIDENCE_THRESHOLD
            and not self.summary.startswith(OUT_OF_SCOPE_PREFIX)
        ):
            raise ValueError(
                f"summary must start with {OUT_OF_SCOPE_PREFIX!r} when confidence is low"
            )
        return self


class StructuredEstimationResponse(BaseModel):
    result: EstimationResult
    model: str
    provider: str
    prompt_version: str
    latency_ms: int = 0
    usage: TokenUsage
    cost_usd: float = 0.0
    project_metadata: dict[str, object] | None = None
    cache_hit: bool = False


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
