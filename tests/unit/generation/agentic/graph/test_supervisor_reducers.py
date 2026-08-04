"""Unit tests for keyed supervisor reducers (routing + contributions)."""

from __future__ import annotations

from app.generation.agentic.graph.supervisor_state import (
    append_contributions,
    append_routing,
    privilege_violations,
)


def test_append_routing_merges_same_step():
    existing = [
        {
            "step": 0,
            "next_agent": "requirements_extractor",
            "reason": "a",
            "source": "llm",
        }
    ]
    new = [
        {
            "step": 0,
            "next_agent": "requirements_extractor",
            "reason": "a-retry",
            "source": "llm",
        }
    ]
    merged = append_routing(existing, new)
    assert len(merged) == 1
    assert merged[0]["reason"] == "a-retry"


def test_append_routing_keeps_distinct_steps():
    merged = append_routing(
        [
            {
                "step": 0,
                "next_agent": "requirements_extractor",
                "reason": "a",
                "source": "llm",
            }
        ],
        [
            {
                "step": 1,
                "next_agent": "budget_searcher",
                "reason": "b",
                "source": "fallback",
            }
        ],
    )
    assert [r["step"] for r in merged] == [0, 1]


def test_append_contributions_keys_include_args_digest():
    base = {
        "step": 1,
        "agent": "budget_searcher",
        "action": "tool:search_budgets",
        "tool": "search_budgets",
        "outcome": "ok",
        "summary": "one",
    }
    a = {**base, "args_digest": "aaa"}
    b = {**base, "args_digest": "bbb"}
    merged = append_contributions([a], [b])
    assert len(merged) == 2

    retry = {**a, "summary": "one-retry"}
    merged2 = append_contributions(merged, [retry])
    assert len(merged2) == 2
    assert any(c["summary"] == "one-retry" for c in merged2)


def test_privilege_violations_filters_denied():
    state = {
        "agent_contributions": [
            {"outcome": "ok", "agent": "a"},
            {"outcome": "denied", "agent": "b"},
            {"outcome": "error", "agent": "c"},
        ]
    }
    denied = privilege_violations(state)
    assert len(denied) == 1
    assert denied[0]["agent"] == "b"
