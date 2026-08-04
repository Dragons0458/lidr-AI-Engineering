"""Unit tests for supervisor node bodies with fake deps (no network)."""

from __future__ import annotations

from typing import Any

import pytest

from app.generation.agentic.agent_schemas import (
    AgentModuleNode,
    AgentStructure,
    AgentTaskNode,
)
from app.generation.agentic.graph.supervisor_nodes import (
    SupervisorDecision,
    SupervisorDeps,
    make_supervisor_nodes,
)
from app.generation.rag.schemas import EstimationQuery


def _brief() -> EstimationQuery:
    return EstimationQuery(
        function="supplier portal",
        technologies=["python"],
        sector="logistics",
        scale="small",
        country="ES",
        regulations=[],
        constraints=[],
    )


def _structure() -> AgentStructure:
    return AgentStructure(
        modules=[
            AgentModuleNode(
                name="Backend",
                tasks=[
                    AgentTaskNode(name="API", description="REST API"),
                    AgentTaskNode(name="Auth", description="Login"),
                ],
            )
        ],
        confidence="medium",
        reasoning="two clear tasks",
    )


async def _fake_reformulate(_transcript: str) -> EstimationQuery:
    return _brief()


async def _fake_structure(_brief: EstimationQuery) -> AgentStructure:
    return _structure()


async def _fake_backend(args: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": 1,
            "budget_id": "b1",
            "estimated_hours": 40.0,
            "distance": 0.1,
            "content": "api work",
        }
    ]


async def _fake_route(_digest: str) -> SupervisorDecision:
    return SupervisorDecision(
        next_agent="requirements_extractor",
        reason="start",
        confidence="high",
    )


def _deps(**overrides: Any) -> SupervisorDeps:
    base = dict(
        reformulate=_fake_reformulate,
        propose_structure=_fake_structure,
        retrieval_backend=_fake_backend,
        route_with_model=_fake_route,
        confidence_threshold=0.6,
        min_grounded_ratio=0.5,
        out_of_range_factor=2.0,
        max_steps=8,
        privilege_strict=False,
    )
    base.update(overrides)
    return SupervisorDeps(**base)


@pytest.mark.asyncio
async def test_requirements_extractor_produces_requirements_and_components():
    nodes = make_supervisor_nodes(_deps())
    update = await nodes["requirements_extractor"](
        {"transcript": "hello " * 30, "estimation_id": "e1", "supervisor_steps": 1}
    )
    assert len(update["requirements"]) >= 1
    assert len(update["components"]) == 2
    assert len(update["agent_contributions"]) == 2
    assert all(c["tool"] is None for c in update["agent_contributions"])


@pytest.mark.asyncio
async def test_budget_searcher_sets_search_completed_and_matches():
    nodes = make_supervisor_nodes(_deps())
    state = {
        "estimation_id": "e1",
        "supervisor_steps": 2,
        "project_brief": _brief().model_dump(mode="json"),
        "components": [
            {
                "component_id": "c1",
                "name": "API",
                "category": "Backend",
                "description": "REST",
            }
        ],
    }
    update = await nodes["budget_searcher"](state)
    assert update["search_completed"] is True
    assert len(update["budget_matches"]) == 1
    assert update["budget_matches"][0]["component_id"] == "c1"


@pytest.mark.asyncio
async def test_estimate_generator_builds_hours_estimate():
    nodes = make_supervisor_nodes(_deps())
    state = {
        "estimation_id": "e1",
        "supervisor_steps": 3,
        "components": [
            {
                "component_id": "c1",
                "name": "API",
                "category": "Backend",
                "description": "",
            }
        ],
        "budget_matches": [
            {
                "component_id": "c1",
                "chunk_id": 1,
                "reference_budget_id": "b1",
                "amount": 40.0,
                "distance": 0.1,
            }
        ],
    }
    update = await nodes["estimate_generator"](state)
    assert update["estimate"]["total_hours"] > 0
    assert update["estimate"]["components"][0]["estimated_hours"] > 0
    assert update["component_anchors"]


@pytest.mark.asyncio
async def test_coherence_validator_publishes_facts():
    nodes = make_supervisor_nodes(_deps())
    state = {
        "estimation_id": "e1",
        "supervisor_steps": 4,
        "components": [
            {
                "component_id": "c1",
                "name": "API",
                "category": "Backend",
                "description": "",
            }
        ],
        "budget_matches": [
            {
                "component_id": "c1",
                "chunk_id": 1,
                "reference_budget_id": "b1",
                "amount": 40.0,
                "distance": 0.1,
            }
        ],
        "estimate": {
            "components": [
                {
                    "name": "API",
                    "estimated_hours": 46.0,
                    "cited_chunk_ids": [1],
                    "rationale": "ok",
                }
            ],
            "total_hours": 46.0,
            "assumptions": [],
            "confidence": "high",
        },
    }
    update = await nodes["coherence_validator"](state)
    assert "confidence" in update
    assert "out_of_range" in update
    assert update["grounded_components"] == 1
    assert update["validation"]["ok"] is True


@pytest.mark.asyncio
async def test_human_review_gate_skips_when_signals_clear():
    nodes = make_supervisor_nodes(_deps(confidence_threshold=0.1))
    state = {
        "components": [
            {
                "component_id": "c1",
                "name": "API",
                "category": "Backend",
                "description": "",
            }
        ],
        "budget_matches": [
            {
                "component_id": "c1",
                "chunk_id": 1,
                "reference_budget_id": "b1",
                "amount": 40.0,
                "distance": 0.1,
            }
        ],
        "estimate": {
            "components": [
                {
                    "name": "API",
                    "estimated_hours": 46.0,
                    "cited_chunk_ids": [1],
                    "rationale": "ok",
                }
            ],
            "total_hours": 46.0,
            "assumptions": [],
            "confidence": "high",
        },
        "confidence": 0.9,
        "validation": {"ok": True, "issues": []},
    }
    update = await nodes["human_review_gate"](state)
    assert update["needs_human_review"] is False


@pytest.mark.asyncio
async def test_human_review_gate_interrupts_when_needed():
    nodes = make_supervisor_nodes(_deps(confidence_threshold=0.95))
    state = {
        "estimation_id": "e1",
        "supervisor_steps": 5,
        "components": [
            {
                "component_id": "c1",
                "name": "API",
                "category": "Backend",
                "description": "",
            }
        ],
        "budget_matches": [],
        "estimate": {
            "components": [
                {
                    "name": "API",
                    "estimated_hours": 0.0,
                    "cited_chunk_ids": [],
                    "rationale": "none",
                }
            ],
            "total_hours": 0.0,
            "assumptions": [],
            "confidence": "low",
        },
        "confidence": 0.1,
        "validation": {"ok": False, "issues": ["x"]},
    }
    with pytest.raises(RuntimeError):
        await nodes["human_review_gate"](state)
