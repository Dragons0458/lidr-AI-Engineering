import pytest
from pydantic import ValidationError

from app.domain.schemas.critic import CriticFeedback, CriticIssue


def test_needs_iteration_requires_blocking_issue() -> None:
    with pytest.raises(ValidationError):
        CriticFeedback(
            verdict="needs_iteration",
            issues=[
                CriticIssue(
                    category="math_error",
                    severity="minor",
                    field_path="total_hours",
                    description="Minor rounding difference only.",
                )
            ],
            confidence_in_review=80,
        )


def test_reject_requires_issues() -> None:
    with pytest.raises(ValidationError):
        CriticFeedback(verdict="reject", issues=[], confidence_in_review=90)
