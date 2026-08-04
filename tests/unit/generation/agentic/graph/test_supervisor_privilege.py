"""Unit tests for supervisor privilege enforcement and audit trail."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.generation.agentic.graph.supervisor_privilege import (
    AGENT_PRIVILEGES,
    PrivilegeViolation,
    assert_allowed,
    guarded_dispatch,
    record_model_action,
)


def test_estimate_generator_holds_calculate_estimate():
    assert "calculate_estimate" in AGENT_PRIVILEGES["estimate_generator"]
    assert "derive_task_hours" not in AGENT_PRIVILEGES["estimate_generator"]


def test_assert_allowed_raises_on_violation():
    with pytest.raises(PrivilegeViolation):
        assert_allowed("budget_searcher", "validate_estimate")


@pytest.mark.asyncio
async def test_guarded_dispatch_denies_before_execute():
    with patch(
        "app.generation.agentic.graph.supervisor_privilege.dispatch_tool",
        new_callable=AsyncMock,
    ) as dispatch:
        envelope, contribution = await guarded_dispatch(
            "budget_searcher",
            "validate_estimate",
            {"components": [], "total_hours": 0},
            step=1,
            estimation_id="e1",
        )
    dispatch.assert_not_called()
    assert envelope["ok"] is False
    assert envelope["error"] == "privilege_denied"
    assert contribution["outcome"] == "denied"


@pytest.mark.asyncio
async def test_guarded_dispatch_strict_raises():
    with pytest.raises(PrivilegeViolation):
        await guarded_dispatch(
            "requirements_extractor",
            "search_budgets",
            {"query": "x", "filters": None},
            step=0,
            privilege_strict=True,
        )


@pytest.mark.asyncio
async def test_guarded_dispatch_allows_and_audits():
    fake = AsyncMock(return_value={"ok": True, "summary": "done", "total_hours": 10})
    with patch(
        "app.generation.agentic.graph.supervisor_privilege.dispatch_tool",
        fake,
    ):
        envelope, contribution = await guarded_dispatch(
            "estimate_generator",
            "calculate_estimate",
            {"components": [{"name": "a", "reference_amounts": [10]}]},
            step=2,
            estimation_id="e2",
        )
    fake.assert_awaited_once()
    assert envelope["ok"] is True
    assert contribution["outcome"] == "ok"
    assert contribution["tool"] == "calculate_estimate"
    assert contribution["args_digest"]


def test_record_model_action_has_no_tool():
    row = record_model_action(
        "requirements_extractor",
        "extract_requirements",
        step=1,
        summary="3 requirements",
        estimation_id="e1",
    )
    assert row["tool"] is None
    assert row["outcome"] == "ok"
