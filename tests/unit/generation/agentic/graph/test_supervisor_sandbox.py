"""Unit tests for Session 14 write sandboxing."""

from __future__ import annotations

from typing import Any

import pytest
import structlog.testing
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.generation.agentic.agent_schemas import (
    AgentModuleNode,
    AgentStructure,
    AgentTaskNode,
)
from app.generation.agentic.graph.supervisor_build import build_supervisor_graph
from app.generation.agentic.graph.supervisor_nodes import (
    SupervisorDecision,
    SupervisorDeps,
)
from app.generation.agentic.graph.supervisor_privilege import redact_args
from app.generation.agentic.graph.supervisor_sandbox import (
    AGENT_TOOL_GRANTS,
    PERSISTED,
    ActionRequest,
    GrantVerificationError,
    ToolRisk,
    execute_guarded,
    guard_action,
    record_intent_sink,
    verify_tool_grants,
)
from app.generation.rag.schemas import EstimationQuery


def test_verify_tool_grants_accepts_real_table():
    verify_tool_grants()


def test_verify_tool_grants_rejects_unknown_risk(monkeypatch):
    monkeypatch.setitem(
        AGENT_TOOL_GRANTS,
        "rogue",
        frozenset({"not_a_real_tool"}),
    )
    with pytest.raises(GrantVerificationError):
        verify_tool_grants()


def test_verify_tool_grants_rejects_unknown_tool():
    with pytest.raises(GrantVerificationError):
        verify_tool_grants(known_tools={"search_budgets"})


def test_guard_action_denies_ungranted_tool():
    decision = guard_action(
        ActionRequest(
            agent="budget_searcher",
            tool="save_estimate",
            args={"estimate": {"total_hours": 1}},
            estimation_id="e1",
            step=1,
        ),
        {"estimation_id": "e1"},
    )
    assert decision.allowed is False


def test_guard_action_denies_mismatched_estimation_id():
    decision = guard_action(
        ActionRequest(
            agent="persistence_agent",
            tool="save_estimate",
            args={"estimation_id": "e1", "estimate": {"total_hours": 1}},
            estimation_id="other",
            step=1,
        ),
        {"estimation_id": "e1"},
    )
    assert decision.allowed is False


def test_guard_action_denies_mismatched_args_estimation_id():
    decision = guard_action(
        ActionRequest(
            agent="persistence_agent",
            tool="save_estimate",
            args={"estimation_id": "other", "estimate": {"total_hours": 1}},
            estimation_id="e1",
            step=1,
        ),
        {"estimation_id": "e1"},
    )
    assert decision.allowed is False


def test_guard_action_denies_empty_estimate():
    decision = guard_action(
        ActionRequest(
            agent="persistence_agent",
            tool="save_estimate",
            args={"estimation_id": "e1", "estimate": {}},
            estimation_id="e1",
            step=1,
        ),
        {"estimation_id": "e1"},
    )
    assert decision.allowed is False


def test_guard_action_defers_irreversible_without_approval():
    decision = guard_action(
        ActionRequest(
            agent="persistence_agent",
            tool="save_estimate",
            args={"estimation_id": "e1", "estimate": {"total_hours": 10}},
            estimation_id="e1",
            step=1,
        ),
        {"estimation_id": "e1"},
    )
    assert decision.allowed is True
    assert decision.requires_human_approval is True
    assert decision.risk == ToolRisk.IRREVERSIBLE


def test_guard_action_clears_with_approval():
    decision = guard_action(
        ActionRequest(
            agent="persistence_agent",
            tool="save_estimate",
            args={"estimation_id": "e1", "estimate": {"total_hours": 10}},
            estimation_id="e1",
            step=1,
        ),
        {"estimation_id": "e1", "human_decision": {"action": "approve"}},
    )
    assert decision.allowed is True
    assert decision.requires_human_approval is False


@pytest.mark.asyncio
async def test_execute_guarded_audits_denied():
    with structlog.testing.capture_logs() as logs:
        envelope, contribution = await execute_guarded(
            ActionRequest(
                agent="budget_searcher",
                tool="save_estimate",
                args={"estimate": {"total_hours": 1}},
                estimation_id="e1",
                step=1,
            ),
            {"estimation_id": "e1"},
        )
    assert envelope["ok"] is False
    assert contribution["outcome"] == "denied"
    assert any(row["event"] == "agent_privilege_denied" for row in logs)


@pytest.mark.asyncio
async def test_execute_guarded_audits_deferred():
    with structlog.testing.capture_logs() as logs:
        envelope, contribution = await execute_guarded(
            ActionRequest(
                agent="persistence_agent",
                tool="save_estimate",
                args={"estimation_id": "e1", "estimate": {"total_hours": 10}},
                estimation_id="e1",
                step=2,
            ),
            {"estimation_id": "e1"},
        )
    assert contribution["outcome"] == "deferred"
    assert any(row["event"] == "agent_action_deferred" for row in logs)


@pytest.mark.asyncio
async def test_execute_guarded_ok_with_sink():
    PERSISTED.clear()
    envelope, contribution = await execute_guarded(
        ActionRequest(
            agent="persistence_agent",
            tool="save_estimate",
            args={"estimation_id": "e1", "estimate": {"total_hours": 10}},
            estimation_id="e1",
            step=2,
        ),
        {"estimation_id": "e1", "human_decision": {"action": "approve"}},
        sink=record_intent_sink,
    )
    assert envelope["ok"] is True
    assert contribution["outcome"] == "ok"
    assert "e1" in PERSISTED


@pytest.mark.asyncio
async def test_execute_guarded_sink_error_is_soft():
    def boom(_eid, _estimate):
        raise RuntimeError("disk full")

    envelope, contribution = await execute_guarded(
        ActionRequest(
            agent="persistence_agent",
            tool="save_estimate",
            args={"estimation_id": "e1", "estimate": {"total_hours": 10}},
            estimation_id="e1",
            step=2,
        ),
        {"estimation_id": "e1", "human_decision": {"action": "approve"}},
        sink=boom,
    )
    assert contribution["outcome"] == "error"
    assert envelope["ok"] is False


def test_redact_args_masks_and_truncates():
    redacted = redact_args(
        {
            "transcript": "secret meeting notes",
            "note": "private",
            "other": "x" * 100,
        }
    )
    assert redacted["transcript"] == "«redacted»"
    assert redacted["note"] == "«redacted»"
    assert redacted["other"].endswith("…")
    assert len(redacted["other"]) == 80


def _persist_deps(**overrides: Any) -> SupervisorDeps:
    async def reformulate(_transcript: str) -> EstimationQuery:
        return EstimationQuery(
            function="portal",
            technologies=["python"],
            sector="logistics",
        )

    async def propose_structure(_brief: EstimationQuery) -> AgentStructure:
        return AgentStructure(
            modules=[
                AgentModuleNode(
                    name="Backend",
                    tasks=[AgentTaskNode(name="API", description="REST")],
                )
            ],
            confidence="high",
            reasoning="ok",
        )

    async def backend(args: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": 1,
                "budget_id": "b1",
                "estimated_hours": 40.0,
                "distance": 0.1,
                "content": "api work",
            }
        ]

    async def route(_digest: str) -> SupervisorDecision:
        raise RuntimeError("use fallback ladder")

    base = dict(
        reformulate=reformulate,
        propose_structure=propose_structure,
        retrieval_backend=backend,
        route_with_model=route,
        confidence_threshold=0.6,
        min_grounded_ratio=0.5,
        out_of_range_factor=2.0,
        max_steps=8,
        privilege_strict=False,
        grounding_max_distance=0.55,
        persistence_enabled=True,
        save_sink=record_intent_sink,
    )
    base.update(overrides)
    return SupervisorDeps(**base)


@pytest.mark.asyncio
async def test_persistence_graph_defers_then_persists_on_approve():
    PERSISTED.clear()
    deps = _persist_deps()
    graph = build_supervisor_graph(deps, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "s14:persist-test"}}
    transcript = (
        "Supplier portal with login, REST API, ERP sync and basic reports for "
        "logistics operations across Spain. Standard enterprise CRUD work. "
    )
    await graph.ainvoke(
        {"transcript": transcript, "estimation_id": "persist-test"},
        config,
    )
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("human_review_gate",)
    reasons = snapshot.interrupts[0].value["reasons"]
    assert "irreversible_write_pending" in reasons
    assert "persist-test" not in PERSISTED

    await graph.ainvoke(
        Command(resume={"action": "approve", "decision": "approve"}),
        config,
    )
    final = await graph.aget_state(config)
    assert not final.next
    assert final.values["saved"]["ok"] is True
    assert "persist-test" in PERSISTED


@pytest.mark.asyncio
async def test_persistence_reject_leaves_deferred():
    PERSISTED.clear()
    deps = _persist_deps()
    graph = build_supervisor_graph(deps, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "s14:persist-reject"}}
    transcript = (
        "Supplier portal with login, REST API, ERP sync and basic reports for "
        "logistics operations across Spain. Standard enterprise CRUD work. "
    )
    await graph.ainvoke(
        {"transcript": transcript, "estimation_id": "persist-reject"},
        config,
    )
    await graph.ainvoke(
        Command(resume={"action": "reject", "decision": "reject"}),
        config,
    )
    final = await graph.aget_state(config)
    assert final.values["saved"]["ok"] is False
    assert final.values["agent_contributions"][-1]["outcome"] == "deferred"
    assert "persist-reject" not in PERSISTED


def test_graph_without_persistence_lacks_persistence_agent():
    deps = _persist_deps(persistence_enabled=False)
    graph = build_supervisor_graph(deps, checkpointer=MemorySaver())
    nodes = set(graph.get_graph().nodes)
    assert "persistence_agent" not in nodes
