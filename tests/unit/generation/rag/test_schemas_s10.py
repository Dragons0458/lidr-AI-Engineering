"""Session 10 schema defaults and task-hours request validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.generation.rag.schemas import (
    RetrievedChunk,
    TaskHoursModuleInput,
    TaskHoursRequest,
    TaskItem,
)


def test_retrieved_chunk_defaults_are_backward_compatible():
    chunk = RetrievedChunk(
        id=1,
        content="OAuth backend",
        chunk_type="budget_component",
        distance=0.3,
    )
    assert chunk.sector == "unknown"
    assert chunk.project_year == 0
    assert chunk.collection == "budget"
    assert chunk.source_id is None
    assert chunk.relevance_score is None
    assert chunk.document_date is None
    assert chunk.estimated_hours is None


def test_task_item_allows_null_engineer_days():
    task = TaskItem(name="Login flow", description="OAuth integration")
    assert task.engineer_days is None


def test_task_hours_request_requires_at_least_one_module():
    with pytest.raises(ValidationError):
        TaskHoursRequest(modules=[])


def test_task_hours_request_accepts_modules_with_tasks():
    payload = TaskHoursRequest(
        modules=[
            TaskHoursModuleInput(
                name="Auth",
                tasks=[{"name": "OAuth login", "description": "Google SSO"}],
            )
        ]
    )
    assert payload.modules[0].name == "Auth"
    assert len(payload.modules[0].tasks) == 1
