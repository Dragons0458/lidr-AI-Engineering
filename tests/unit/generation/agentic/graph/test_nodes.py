"""Unit tests for Session 13 graph nodes with injectable fakes."""

from __future__ import annotations

import asyncio
import copy

import logfire
import pytest

from app.generation.agentic.agent_schemas import (
    AgentModuleNode,
    AgentStructure,
    AgentTaskNode,
    SearchBudgetsArgs,
)
from app.generation.agentic.graph.nodes import GraphNodeDeps, make_graph_nodes
from app.generation.rag.schemas import EstimationQuery

logfire.configure(send_to_logfire=False)


def _deps(*, backend=None, structure=None) -> GraphNodeDeps:
    async def reformulate(transcript: str) -> EstimationQuery:
        assert "portal" in transcript
        return EstimationQuery(
            function="Customer portal",
            technologies=["Django"],
            constraints=["GDPR"],
            regulations=["LOPD"],
        )

    async def propose_structure(brief: EstimationQuery) -> AgentStructure:
        if structure is not None:
            return structure
        return AgentStructure(
            modules=[
                AgentModuleNode(
                    name="Core",
                    tasks=[
                        AgentTaskNode(name="Auth", description="OAuth login"),
                        AgentTaskNode(name="Dashboard", description="KPIs"),
                    ],
                ),
                AgentModuleNode(
                    name="Integrations",
                    tasks=[AgentTaskNode(name="Auth", description="SSO bridge")],
                ),
            ],
            confidence="high",
            reasoning="clear brief",
        )

    async def default_backend(args: SearchBudgetsArgs) -> list[dict]:
        return [
            {
                "id": 11,
                "budget_id": "B-1",
                "estimated_hours": 40,
                "distance": 0.2,
            },
            {
                "id": 12,
                "budget_id": "B-2",
                "estimated_hours": 60,
                "distance": 0.3,
            },
        ]

    return GraphNodeDeps(
        reformulate=reformulate,
        propose_structure=propose_structure,
        retrieval_backend=backend or default_backend,
    )


@pytest.mark.asyncio
async def test_extract_requirements_returns_partial_update_only():
    nodes = make_graph_nodes(_deps())
    state = {"transcript": "Build a customer portal with GDPR"}
    original = copy.deepcopy(state)
    update = await nodes["extract_requirements"](state)
    assert set(update) == {"project_brief", "requirements"}
    assert state == original
    assert "function: Customer portal" in update["requirements"]
    assert "technology: Django" in update["requirements"]
    assert "constraint: GDPR" in update["requirements"]
    assert "regulation: LOPD" in update["requirements"]


@pytest.mark.asyncio
async def test_classify_components_unique_ids_for_same_task_name():
    nodes = make_graph_nodes(_deps())
    update = await nodes["classify_components"](
        {"project_brief": EstimationQuery(function="Portal").model_dump(mode="json")}
    )
    assert set(update) == {"components"}
    ids = [c["component_id"] for c in update["components"]]
    assert len(ids) == len(set(ids))
    assert len(update["components"]) == 3
    assert update["components"][0]["category"] == "Core"
    assert update["components"][2]["category"] == "Integrations"


@pytest.mark.asyncio
async def test_search_budgets_is_strictly_sequential():
    order: list[str] = []
    in_flight = 0
    max_in_flight = 0

    async def backend(args: SearchBudgetsArgs) -> list[dict]:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        order.append(args.query)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return [
            {
                "id": len(order),
                "budget_id": f"B-{len(order)}",
                "estimated_hours": 10 * len(order),
                "distance": 0.1,
            }
        ]

    nodes = make_graph_nodes(_deps(backend=backend))
    components = [
        {
            "component_id": "m0-t0-auth",
            "name": "Auth",
            "category": "Core",
            "description": "login",
        },
        {
            "component_id": "m0-t1-dash",
            "name": "Dashboard",
            "category": "Core",
            "description": "kpis",
        },
        {
            "component_id": "m1-t0-auth",
            "name": "Auth",
            "category": "Integrations",
            "description": "sso",
        },
    ]
    update = await nodes["search_budgets"](
        {
            "components": components,
            "project_brief": {"function": "Portal", "sector": None},
        }
    )
    assert max_in_flight == 1
    assert "gather" not in order
    assert len(order) == 3
    assert all(
        m["component_id"] in {c["component_id"] for c in components}
        for m in update["budget_matches"]
    )
    assert update["budget_matches"][0]["component_id"] == "m0-t0-auth"
    assert update["budget_matches"][2]["component_id"] == "m1-t0-auth"


@pytest.mark.asyncio
async def test_search_budgets_empty_results_do_not_fail():
    async def backend(args: SearchBudgetsArgs) -> list[dict]:
        return []

    nodes = make_graph_nodes(_deps(backend=backend))
    update = await nodes["search_budgets"](
        {
            "components": [
                {
                    "component_id": "c1",
                    "name": "X",
                    "category": "M",
                    "description": "",
                }
            ],
            "project_brief": {"function": "Portal"},
        }
    )
    assert update == {"budget_matches": []}


@pytest.mark.asyncio
async def test_generate_estimate_median_contingency_and_unbudgeted():
    nodes = make_graph_nodes(_deps())
    components = [
        {
            "component_id": "c1",
            "name": "Auth",
            "category": "Core",
            "description": "",
        },
        {
            "component_id": "c2",
            "name": "Mystery",
            "category": "Core",
            "description": "",
        },
    ]
    matches = [
        {
            "component_id": "c1",
            "chunk_id": 1,
            "reference_budget_id": "B1",
            "amount": 40.0,
            "distance": 0.1,
        },
        {
            "component_id": "c1",
            "chunk_id": 2,
            "reference_budget_id": "B2",
            "amount": 60.0,
            "distance": 0.2,
        },
    ]
    update = await nodes["generate_estimate"](
        {"components": components, "budget_matches": matches}
    )
    estimate = update["estimate"]
    # median(40,60)=50 * 1.15 = 57.5; unbudgeted = 0
    assert estimate["total_hours"] == 57.5
    assert estimate["components"][0]["estimated_hours"] == 57.5
    assert estimate["components"][1]["estimated_hours"] == 0.0
    assert any("Mystery" in a for a in estimate["assumptions"])
    assert estimate["components"][0]["cited_chunk_ids"] == [1, 2]


@pytest.mark.asyncio
async def test_validate_and_consolidate_validated_vs_needs_review():
    nodes = make_graph_nodes(_deps())
    components = [
        {
            "component_id": "c1",
            "name": "Auth",
            "category": "Core",
            "description": "",
        }
    ]
    matches = [
        {
            "component_id": "c1",
            "chunk_id": 1,
            "reference_budget_id": "B1",
            "amount": 40.0,
            "distance": 0.1,
        },
        {
            "component_id": "c1",
            "chunk_id": 2,
            "reference_budget_id": "B2",
            "amount": 60.0,
            "distance": 0.2,
        },
    ]
    generated = await nodes["generate_estimate"](
        {"components": components, "budget_matches": matches}
    )
    ok = await nodes["validate_and_consolidate"](
        {
            "components": components,
            "budget_matches": matches,
            "estimate": generated["estimate"],
        }
    )
    assert ok == {"status": "validated"}

    bad = await nodes["validate_and_consolidate"](
        {
            "components": components,
            "budget_matches": [],
            "estimate": {
                "components": [
                    {
                        "name": "Auth",
                        "estimated_hours": 0.0,
                        "cited_chunk_ids": [],
                        "rationale": "none",
                    }
                ],
                "total_hours": 0.0,
                "assumptions": [],
                "confidence": "low",
            },
        }
    )
    assert bad["status"] == "needs_review"
    assert bad["errors"]
    assert all(isinstance(e, str) for e in bad["errors"])
