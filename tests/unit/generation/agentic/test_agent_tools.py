"""Unit tests for the Session 12 agent tools (no network, no DB)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.generation.agentic.agent_schemas import SearchBudgetsArgs
from app.generation.agentic.agent_tools import (
    CONTINGENCY_FACTOR,
    DERIVE_TASK_HOURS_TOOL,
    adapt_legacy_backend,
    adapt_recovery_backend,
    calculate_estimate,
    derive_task_hours,
    dispatch_recovery_tool,
    dispatch_tool,
    search_budgets,
    validate_estimate,
)


# --- calculate_estimate ---------------------------------------------------- #
def test_calculate_estimate_median_plus_contingency():
    result = calculate_estimate(
        {
            "components": [
                {"name": "Auth backend", "reference_amounts": [100.0, 200.0, 300.0]}
            ]
        }
    )
    # median(100,200,300)=200; +15% contingency => 230.
    assert result["components"][0]["estimated_hours"] == pytest.approx(
        200 * (1 + CONTINGENCY_FACTOR)
    )
    assert result["total_hours"] == pytest.approx(230.0)
    assert result["components"][0]["unbudgeted"] is False


def test_calculate_estimate_flags_unbudgeted_without_inventing_hours():
    result = calculate_estimate(
        {"components": [{"name": "Mystery module", "reference_amounts": []}]}
    )
    component = result["components"][0]
    assert component["estimated_hours"] == 0.0
    assert component["unbudgeted"] is True
    assert result["total_hours"] == 0.0


def test_calculate_estimate_sums_components():
    result = calculate_estimate(
        {
            "components": [
                {"name": "A", "reference_amounts": [100.0]},
                {"name": "B", "reference_amounts": [200.0]},
            ]
        }
    )
    assert result["total_hours"] == pytest.approx(115.0 + 230.0)


def test_calculate_estimate_rejects_bad_args():
    with pytest.raises(ValidationError):
        calculate_estimate({"components": [{"name": "A"}]})  # missing reference_amounts


# --- validate_estimate ----------------------------------------------------- #
def test_validate_estimate_passes_clean_estimate():
    result = validate_estimate(
        {
            "components": [
                {"name": "A", "estimated_hours": 115.0, "reference_amounts": [100.0]}
            ],
            "total_hours": 115.0,
        }
    )
    assert result["ok"] is True
    assert result["issues"] == []


def test_validate_estimate_flags_unbudgeted_and_total_mismatch():
    result = validate_estimate(
        {
            "components": [
                {"name": "A", "estimated_hours": 50.0, "reference_amounts": []}
            ],
            "total_hours": 999.0,
        }
    )
    assert result["ok"] is False
    joined = " ".join(result["issues"]).lower()
    assert "no historical reference" in joined
    assert "does not match" in joined


def test_validate_estimate_flags_out_of_range_component():
    # reference 100 → plausible range [50, 200]; 1000 is out of range.
    result = validate_estimate(
        {
            "components": [
                {"name": "A", "estimated_hours": 1000.0, "reference_amounts": [100.0]}
            ],
            "total_hours": 1000.0,
        }
    )
    assert result["ok"] is False
    assert any("outside the plausible range" in issue for issue in result["issues"])


def test_validate_estimate_flags_nonpositive_total():
    result = validate_estimate({"components": [], "total_hours": 0.0})
    assert result["ok"] is False
    assert any("non-positive" in issue for issue in result["issues"])


# --- search_budgets + dispatch --------------------------------------------- #
async def test_search_budgets_uses_injected_backend():
    async def fake_backend(args: SearchBudgetsArgs) -> list[dict]:
        assert args.query == "auth backend"
        return [
            {"id": 1, "estimated_hours": 420.0, "content_preview": "x", "distance": 0.1}
        ]

    result = await search_budgets(
        {"query": "auth backend", "filters": None}, backend=fake_backend
    )
    assert result["count"] == 1
    assert "420.0" in result["summary"]


async def test_search_budgets_rejects_empty_query():
    async def fake_backend(args: SearchBudgetsArgs) -> list[dict]:
        return []

    with pytest.raises(ValidationError):
        await search_budgets({"query": "", "filters": None}, backend=fake_backend)


async def test_dispatch_unknown_tool_raises():
    async def fake_backend(args: SearchBudgetsArgs) -> list[dict]:
        return []

    with pytest.raises(ValueError, match="Unknown tool"):
        await dispatch_tool("nonexistent", {}, backend=fake_backend)


def test_derive_task_hours_schema_is_strict_at_every_object_level():
    assert DERIVE_TASK_HOURS_TOOL["strict"] is True
    parameters = DERIVE_TASK_HOURS_TOOL["parameters"]
    assert parameters["additionalProperties"] is False
    assert (
        parameters["properties"]["neighbors"]["items"]["additionalProperties"] is False
    )
    assert set(parameters["required"]) == {"task_ref", "module", "task", "neighbors"}


def test_derive_task_hours_uses_injected_consensus_and_preserves_provenance():
    captured = {}

    def consensus(neighbors):
        captured["neighbors"] = neighbors
        return 45, 0.88, 0.12

    result = derive_task_hours(
        {
            "task_ref": "task-0",
            "module": "Core",
            "task": "Build",
            "neighbors": [
                {
                    "source_id": 7,
                    "budget_id": "BUD-7",
                    "estimated_hours": 40,
                    "distance": 0.08,
                }
            ],
        },
        consensus=consensus,
    )
    assert captured["neighbors"] == [(40, 0.08)]
    assert result["estimated_hours"] == 45
    assert result["reliability"] == 0.88
    assert result["dispersion"] == 0.12
    assert result["neighbors"][0] == {
        "source_id": 7,
        "budget_id": "BUD-7",
        "estimated_hours": 40,
        "distance": 0.08,
    }


def test_derive_task_hours_empty_neighbors_does_not_call_consensus():
    def consensus(_neighbors):
        raise AssertionError("consensus must not run")

    result = derive_task_hours(
        {
            "task_ref": "task-0",
            "module": "Core",
            "task": "Build",
            "neighbors": [],
        },
        consensus=consensus,
    )
    assert result["has_match"] is False
    assert result["estimated_hours"] is None


async def test_recovery_dispatch_and_unknown_tool():
    async def backend(query, sectors):
        assert (query, sectors) == ("auth", ["finance"])
        return [{"id": 1, "estimated_hours": 20}]

    result = await dispatch_recovery_tool(
        "search_budgets",
        {"query": "auth", "sectors": ["finance"]},
        backend=backend,
        consensus=lambda values: (20, 1.0, 0.0),
    )
    assert result["count"] == 1
    with pytest.raises(ValueError, match="Unknown recovery tool"):
        await dispatch_recovery_tool(
            "unknown",
            {},
            backend=backend,
            consensus=lambda values: (20, 1.0, 0.0),
        )


async def test_legacy_and_recovery_backend_adapters_keep_signatures_separate():
    async def recovery(query, sectors):
        return [{"query": query, "sectors": sectors}]

    legacy = adapt_recovery_backend(recovery)
    result = await legacy(
        SearchBudgetsArgs(
            query="payments",
            filters={"sectors": ["finance"], "component_type": None},
        )
    )
    assert result == [{"query": "payments", "sectors": ["finance"]}]

    async def legacy_backend(args):
        assert isinstance(args, SearchBudgetsArgs)
        return [{"query": args.query}]

    recovery_adapter = adapt_legacy_backend(legacy_backend)
    assert await recovery_adapter("mobile", None) == [{"query": "mobile"}]


async def test_dispatch_tool_keeps_legacy_backend_signature():
    seen = {}

    async def legacy_backend(args):
        seen["args"] = args
        return []

    await dispatch_tool(
        "search_budgets",
        {"query": "auth", "filters": None},
        backend=legacy_backend,
    )
    assert isinstance(seen["args"], SearchBudgetsArgs)
