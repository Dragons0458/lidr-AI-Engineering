"""Pure estimate builder helpers for the multi-agent graph."""

from __future__ import annotations

from app.generation.agentic.graph.estimate_builder import (
    build_estimate,
    flag_reason,
    modules_from_structure,
    recompute_estimate_totals,
)


def _modules(*pairs):
    return [
        {"name": "M", "tasks": [{"name": n, "estimated_hours": h} for n, h in pairs]}
    ]


def test_modules_from_structure_flattens_tasks():
    structure = {
        "modules": [
            {
                "name": "Backend",
                "tasks": [{"name": "API", "description": "REST"}],
            }
        ]
    }
    modules = modules_from_structure(structure)
    assert modules == [
        {"name": "Backend", "tasks": [{"name": "API", "description": "REST"}]}
    ]


def test_flag_reason_no_match():
    assert flag_reason({"has_match": False}, reliability_threshold=0.35)


def test_flag_reason_range():
    assert flag_reason(
        {"has_match": True, "hours_range": (10, 20)},
        reliability_threshold=0.35,
    )


def test_flag_reason_low_reliability():
    assert flag_reason(
        {"has_match": True, "reliability": 0.1},
        reliability_threshold=0.35,
    )


def test_flag_reason_none_when_grounded():
    assert (
        flag_reason(
            {"has_match": True, "reliability": 0.9},
            reliability_threshold=0.35,
        )
        is None
    )


def test_recompute_estimate_totals_all_grounded():
    assert recompute_estimate_totals(_modules(("A", 40), ("B", 24))) == {
        "total_engineer_hours": 64.0,
        "total_engineer_days": 8,
        "grounded_task_ratio": 1.0,
        "confidence": "high",
    }


def test_recompute_estimate_totals_mixed():
    totals = recompute_estimate_totals(_modules(("A", 40), ("B", None)))
    assert totals["confidence"] == "medium"
    assert totals["grounded_task_ratio"] == 0.5


def test_recompute_estimate_totals_none_grounded():
    totals = recompute_estimate_totals(_modules(("A", None), ("B", None)))
    assert totals["confidence"] == "low"
    assert totals["grounded_task_ratio"] == 0.0


def test_build_estimate_grafts_hours_and_totals():
    estimate = build_estimate(
        [{"name": "M", "tasks": [{"name": "A", "description": "a"}, {"name": "B"}]}],
        [
            {
                "module": "M",
                "task": "A",
                "estimated_hours": 40,
                "has_match": True,
                "reliability": 0.9,
            }
        ],
    )
    tasks = estimate["modules"][0]["tasks"]
    assert tasks[0]["estimated_hours"] == 40
    assert tasks[1]["estimated_hours"] is None
    assert estimate["confidence"] == "medium"
