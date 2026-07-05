"""Unit tests for Session 12 agent schemas (serialization + trace render)."""

from __future__ import annotations

from app.generation.agentic.agent_schemas import (
    AgentComponent,
    AgentEstimate,
    AgentStep,
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
