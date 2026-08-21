"""Pydantic contracts for the Session 16 production golden set and reports."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

AbstentionSignal = Literal["explicit", "proxy"]
ArmName = Literal["rag", "graph"]
CaseVerdict = Literal["passed", "failed", "throttled", "error", "skipped"]


class GoldenCase(BaseModel):
    """One production-quality case: a transcript, an expected number, a band."""

    id: str = Field(min_length=1)
    title: str
    difficulty: str
    transcript_path: str
    expected_engineer_days: int = Field(ge=0)
    acceptable_range: tuple[int, int]
    expected_sources_include: list[str] = Field(default_factory=list)
    expect_abstention: bool = False
    notes: str = ""

    @field_validator("acceptable_range", mode="before")
    @classmethod
    def coerce_range(cls, value: object) -> tuple[int, int]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
        raise ValueError("acceptable_range must be [low, high]")

    @model_validator(mode="after")
    def range_contains_expected(self) -> GoldenCase:
        low, high = self.acceptable_range
        if low > high:
            raise ValueError(f"{self.id}: range low > high")
        if not (low <= self.expected_engineer_days <= high):
            raise ValueError(
                f"{self.id}: expected_engineer_days "
                f"{self.expected_engineer_days} outside {self.acceptable_range}"
            )
        return self


class GoldenSet(BaseModel):
    """Versioned production golden set. Comparable reports share the same sha256."""

    description: str
    corpus: str
    units: Literal["engineer_days"]
    hours_per_day: int = Field(default=8, ge=1)
    cases: list[GoldenCase] = Field(min_length=5)

    @model_validator(mode="after")
    def unique_ids_and_one_abstention(self) -> GoldenSet:
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("golden set case ids must be unique")
        if not any(case.expect_abstention for case in self.cases):
            raise ValueError("golden set needs at least one expect_abstention case")
        return self


class Outcome(BaseModel):
    """Normalised prediction from either architecture arm."""

    engineer_days: int | None = None
    confidence: str | None = None
    abstained: bool = False
    abstention_signal: AbstentionSignal | None = None
    source_ids: list[str] = Field(default_factory=list)
    assumptions_count: int = 0
    grounded_ratio: float | None = None
    latency_ms: float = 0.0
    llm_calls: int = 0
    http_status: int | None = None
    error: str | None = None
    throttled: bool = False
    skipped: bool = False
    skip_reason: str | None = None
    cost_usd: float | None = None
    cache_hit: bool = False


class CaseEvaluation(BaseModel):
    """Per-case verdict. `passed` is the definition of 'the system was good'."""

    case_id: str
    arm: ArmName
    verdict: CaseVerdict
    passed: bool
    expected_engineer_days: int
    predicted_engineer_days: int | None = None
    acceptable_range: tuple[int, int]
    abs_error: float | None = None
    abstained: bool = False
    abstention_signal: AbstentionSignal | None = None
    source_hit: bool | None = None
    latency_ms: float = 0.0
    llm_calls: int = 0
    http_status: int | None = None
    error: str | None = None
    cost_usd: float | None = None
    notes: str = ""


class ArmReport(BaseModel):
    """Aggregates for one architecture arm."""

    arm: ArmName
    skipped: bool = False
    skip_reason: str | None = None
    n_cases: int = 0
    n_estimation: int = 0
    n_abstention: int = 0
    n_passed: int = 0
    n_failed: int = 0
    n_throttled: int = 0
    n_error: int = 0
    within_range_rate: float | None = None
    mean_absolute_error: float | None = None
    abstention_correct: bool | None = None
    mean_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p95_n: int = 0
    error_rate: float | None = None
    abstention_rate: float | None = None
    source_recall: float | None = None
    total_cost_usd: float | None = None
    mean_cost_usd: float | None = None
    cache_hit_rate: float | None = None
    evaluations: list[CaseEvaluation] = Field(default_factory=list)


class EvalReport(BaseModel):
    """Full harness report. Never contains transcripts, prompts, or API keys."""

    run_id: str
    label: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    golden_set: dict
    environment: dict = Field(default_factory=dict)
    arms: dict[str, ArmReport] = Field(default_factory=dict)
    ab: dict | None = None
