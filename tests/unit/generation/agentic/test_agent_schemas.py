"""Unit tests for Session 12 agent schemas (serialization + trace render)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.schemas.agent_trace import AgentTrace as SharedAgentTrace
from app.generation.agentic.agent_schemas import (
    AgentComponent,
    AgentEstimate,
    AgentModuleNode,
    AgentRecoveryNeighbor,
    AgentStep,
    AgentStructure,
    AgentTaskDerivation,
    AgentTaskHoursRun,
    AgentTaskNode,
    AgentTaskRef,
    AgentTrace,
    SearchBudgetsArgs,
)


def test_search_budgets_args_roundtrip():
    args = SearchBudgetsArgs(query="auth backend", filters=None)
    dumped = args.model_dump()
    restored = SearchBudgetsArgs.model_validate(dumped)
    assert restored.query == "auth backend"
    assert restored.filters is None


def test_agent_trace_render_step_format():
    trace = AgentTrace(
        steps=[
            AgentStep(
                step=1,
                reasoning_summary="Decompose and search.",
                tool="search_budgets",
                tool_args={"query": "auth", "filters": None},
                observation="2 historical items",
            ),
            AgentStep(
                step=2,
                reasoning_summary="Calculate totals.",
                tool="calculate_estimate",
                tool_args={
                    "components": [{"name": "Auth", "reference_amounts": [100.0]}]
                },
                observation="total=115.0h",
            ),
        ]
    )
    rendered = trace.render()
    assert "STEP 1" in rendered
    assert "STEP 2" in rendered
    assert "reasoning:" in rendered
    assert "action:" in rendered
    assert "observation:" in rendered
    assert "search_budgets" in rendered
    assert "calculate_estimate" in rendered


def test_agent_estimate_serialization():
    estimate = AgentEstimate(
        components=[
            AgentComponent(
                name="Auth", estimated_hours=115.0, rationale="median+buffer"
            )
        ],
        total_hours=115.0,
        assumptions=["Rails/Postgres as stated."],
        confidence="medium",
    )
    data = estimate.model_dump()
    restored = AgentEstimate.model_validate(data)
    assert restored.total_hours == 115.0
    assert restored.confidence == "medium"


def test_structure_and_recovery_models_roundtrip():
    structure = AgentStructure(
        modules=[
            AgentModuleNode(
                name="Core",
                tasks=[AgentTaskNode(name="Build", description="Implementation")],
            )
        ],
        confidence="high",
        reasoning="Clear",
    )
    flagged = AgentTaskRef(
        task_ref="task-0",
        module="Core",
        task="Build",
        reason="no historical match",
    )
    derivation = AgentTaskDerivation(
        task_ref=flagged.task_ref,
        module=flagged.module,
        task=flagged.task,
        estimated_hours=20,
        reliability=0.8,
        dispersion=0.1,
        has_match=True,
        neighbors=[
            AgentRecoveryNeighbor(
                source_id=1,
                budget_id="BUD-1",
                estimated_hours=20,
                distance=0.1,
            )
        ],
    )
    run = AgentTaskHoursRun(derivations=[derivation], iterations=2)
    assert AgentStructure.model_validate(structure.model_dump()) == structure
    assert AgentTaskHoursRun.model_validate(run.model_dump()) == run


@pytest.mark.parametrize(
    "changes",
    [
        {"estimated_hours": None},
        {"neighbors": []},
    ],
)
def test_matched_derivation_requires_hours_and_neighbors(changes):
    values = {
        "task_ref": "task-0",
        "module": "Core",
        "task": "Build",
        "estimated_hours": 20,
        "has_match": True,
        "neighbors": [
            {
                "source_id": 1,
                "budget_id": "BUD-1",
                "estimated_hours": 20,
                "distance": 0.1,
            }
        ],
    }
    values.update(changes)
    with pytest.raises(ValidationError, match="requires estimated_hours"):
        AgentTaskDerivation(**values)


def test_shared_trace_reexport_keeps_json_stable():
    assert AgentTrace is SharedAgentTrace
    trace = AgentTrace(
        steps=[
            AgentStep(
                step=1,
                reasoning_summary=None,
                tool="derive_task_hours",
                tool_args={"task_ref": "task-0"},
                observation="completed",
            )
        ]
    )
    assert AgentTrace.model_validate_json(trace.model_dump_json()) == trace
    assert "(no reasoning summary emitted)" in trace.render()
