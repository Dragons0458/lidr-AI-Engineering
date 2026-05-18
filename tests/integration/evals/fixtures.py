from dataclasses import dataclass, field
from typing import Any

try:
    from deepeval.dataset import EvaluationDataset, Golden
except ImportError:

    @dataclass
    class Golden:
        input: str
        expected_output: str | None = None
        additional_metadata: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class EvaluationDataset:
        goldens: list[Golden]


golden_dataset = EvaluationDataset(
    goldens=[
        Golden(
            input="Build a simple landing page with contact form.",
            expected_output=None,
            additional_metadata={
                "category": "small_project",
                "expected_hours_range": (16, 40),
                "expected_components": ["frontend", "form_handling"],
            },
        ),
        Golden(
            input=(
                "We need an internal admin dashboard with user management, "
                "role-based permissions, audit log, and weekly email reports."
            ),
            expected_output=None,
            additional_metadata={
                "category": "medium_project",
                "expected_hours_range": (200, 400),
                "expected_components": [
                    "backend",
                    "frontend",
                    "auth",
                    "email_jobs",
                ],
            },
        ),
        Golden(
            input=(
                "Create a customer portal with login, profile management, "
                "support ticket tracking, and admin reporting."
            ),
            expected_output=None,
            additional_metadata={
                "category": "medium_project",
                "expected_hours_range": (160, 360),
                "expected_components": [
                    "backend",
                    "frontend",
                    "auth",
                    "reporting",
                ],
            },
        ),
        Golden(
            input=(
                "Build a weekly sales report automation that reads database data "
                "and sends summary emails to managers."
            ),
            expected_output=None,
            additional_metadata={
                "category": "small_automation",
                "expected_hours_range": (60, 160),
                "expected_components": ["backend", "database", "email_jobs"],
            },
        ),
    ]
)
