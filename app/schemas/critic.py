from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

CriticIssueCategory = Literal[
    "math_error",
    "hallucination",
    "scope_mismatch",
    "phase_imbalance",
    "missing_assumption",
    "unrealistic_estimate",
    "tier_mismatch",
]
CriticIssueSeverity = Literal["critical", "major", "minor"]
CriticVerdict = Literal["accept", "needs_iteration", "reject"]


class CriticIssue(BaseModel):
    category: CriticIssueCategory
    severity: CriticIssueSeverity
    field_path: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=5, max_length=500)
    suggested_fix: str | None = None


class CriticFeedback(BaseModel):
    verdict: CriticVerdict
    issues: list[CriticIssue] = Field(default_factory=list, max_length=12)
    confidence_in_review: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_verdict_issues(self) -> CriticFeedback:
        has_blocking = any(
            issue.severity in ("critical", "major") for issue in self.issues
        )
        if self.verdict == "needs_iteration" and not has_blocking:
            raise ValueError(
                "needs_iteration requires at least one critical/major issue"
            )
        if self.verdict == "reject" and not self.issues:
            raise ValueError("reject requires at least one issue")
        return self
