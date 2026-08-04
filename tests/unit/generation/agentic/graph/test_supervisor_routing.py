"""Unit tests for supervisor routing brakes (budget, legality, fallback)."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.types import Command

from app.generation.agentic.graph.supervisor_nodes import (
    SupervisorDecision,
    SupervisorDeps,
    make_supervisor_nodes,
)
from app.generation.rag.schemas import EstimationQuery


async def _boom(_digest: str) -> SupervisorDecision:
    raise RuntimeError("router down")


async def _illegal(_digest: str) -> SupervisorDecision:
    return SupervisorDecision(
        next_agent="estimate_generator",
        reason="skip ahead illegally",
        confidence="high",
    )


async def _noop_reformulate(transcript: str) -> EstimationQuery:
    return EstimationQuery(
        function="x",
        technologies=[],
        sector=None,
        scale="unknown",
        country=None,
        regulations=[],
        constraints=[],
    )


async def _noop_structure(_brief: EstimationQuery):
    from app.generation.agentic.agent_schemas import AgentStructure

    return AgentStructure(
        modules=[], confidence="low", reasoning="empty for routing tests"
    )


async def _noop_backend(_args: Any) -> list:
    return []


def _deps(route) -> SupervisorDeps:
    return SupervisorDeps(
        reformulate=_noop_reformulate,
        propose_structure=_noop_structure,
        retrieval_backend=_noop_backend,
        route_with_model=route,
        confidence_threshold=0.6,
        min_grounded_ratio=0.5,
        out_of_range_factor=2.0,
        max_steps=8,
        privilege_strict=False,
    )


@pytest.mark.asyncio
async def test_step_budget_forces_finish():
    nodes = make_supervisor_nodes(_deps(_boom))
    cmd = await nodes["supervisor"](
        {"transcript": "x" * 120, "supervisor_steps": 8, "routing_history": []}
    )
    assert isinstance(cmd, Command)
    assert cmd.goto == "human_review_gate"
    assert cmd.update["routing_history"][0]["source"] == "limit"
    assert cmd.update["routing_history"][0]["next_agent"] == "finish"


@pytest.mark.asyncio
async def test_router_failure_falls_back_to_ladder():
    nodes = make_supervisor_nodes(_deps(_boom))
    cmd = await nodes["supervisor"](
        {"transcript": "x" * 120, "supervisor_steps": 0, "routing_history": []}
    )
    assert cmd.goto == "requirements_extractor"
    assert cmd.update["routing_history"][0]["source"] == "fallback"


@pytest.mark.asyncio
async def test_illegal_proposal_overridden():
    nodes = make_supervisor_nodes(_deps(_illegal))
    cmd = await nodes["supervisor"](
        {"transcript": "x" * 120, "supervisor_steps": 0, "routing_history": []}
    )
    assert cmd.goto == "requirements_extractor"
    assert cmd.update["routing_history"][0]["source"] == "fallback"
    assert "not legal" in cmd.update["routing_history"][0]["reason"]


@pytest.mark.asyncio
async def test_estimate_generator_legal_after_empty_search():
    async def route_to_generator(_digest: str) -> SupervisorDecision:
        return SupervisorDecision(
            next_agent="estimate_generator",
            reason="search done, estimate now",
            confidence="medium",
        )

    nodes = make_supervisor_nodes(_deps(route_to_generator))
    cmd = await nodes["supervisor"](
        {
            "transcript": "x" * 120,
            "supervisor_steps": 2,
            "components": [
                {
                    "component_id": "c1",
                    "name": "A",
                    "category": "m",
                    "description": "",
                }
            ],
            "budget_matches": [],
            "search_completed": True,
            "routing_history": [
                {
                    "step": 0,
                    "next_agent": "requirements_extractor",
                    "reason": "ok",
                    "source": "llm",
                },
                {
                    "step": 1,
                    "next_agent": "budget_searcher",
                    "reason": "ok",
                    "source": "llm",
                },
            ],
        }
    )
    assert cmd.goto == "estimate_generator"
    assert cmd.update["routing_history"][0]["source"] == "llm"


@pytest.mark.asyncio
async def test_already_ran_agent_not_revisited():
    async def route_extractor(_digest: str) -> SupervisorDecision:
        return SupervisorDecision(
            next_agent="requirements_extractor",
            reason="again",
            confidence="low",
        )

    nodes = make_supervisor_nodes(_deps(route_extractor))
    cmd = await nodes["supervisor"](
        {
            "transcript": "x" * 120,
            "supervisor_steps": 1,
            "requirements": ["a"],
            "components": [
                {
                    "component_id": "c1",
                    "name": "A",
                    "category": "m",
                    "description": "",
                }
            ],
            "routing_history": [
                {
                    "step": 0,
                    "next_agent": "requirements_extractor",
                    "reason": "start",
                    "source": "llm",
                }
            ],
        }
    )
    assert cmd.goto == "budget_searcher"
    assert cmd.update["routing_history"][0]["source"] == "fallback"
