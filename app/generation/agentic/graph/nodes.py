"""Five sequential nodes for the Session 13 estimation graph.

Nodes are pure with respect to state: they never mutate the received dict or
lists, and they return only the fields they change. External I/O is injected
via ``GraphNodeDeps`` so unit tests can swap fakes without touching singletons.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import logfire
import structlog
from langgraph.config import get_config

from app.generation.agentic.agent_schemas import (
    AgentComponent,
    AgentEstimate,
    AgentStructure,
    SearchBudgetsArgs,
)
from app.generation.agentic.agent_tools import (
    calculate_estimate,
    search_budgets,
    validate_estimate,
)
from app.generation.agentic.graph.state import (
    BudgetMatch,
    Component,
    EstimationState,
    ensure_unique_component_ids,
    stable_component_id,
)
from app.generation.rag.schemas import EstimationQuery
from app.generation.rag.task_hours import compose_task_search_text

log = structlog.get_logger()

ReformulateFn = Callable[[str], Awaitable[EstimationQuery]]
StructureFn = Callable[[EstimationQuery], Awaitable[AgentStructure]]
LegacyRetrievalBackend = Callable[[SearchBudgetsArgs], Awaitable[list[dict[str, Any]]]]


@dataclass(frozen=True)
class GraphNodeDeps:
    """Injectable collaborators for the five graph nodes."""

    reformulate: ReformulateFn
    propose_structure: StructureFn
    retrieval_backend: LegacyRetrievalBackend


def _estimation_id() -> str:
    """Read ``thread_id`` from the active LangGraph runnable config."""
    try:
        config = get_config()
    except RuntimeError:
        return "unknown"
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    return str(thread_id) if thread_id else "unknown"


def requirements_from_brief(brief: EstimationQuery) -> list[str]:
    """Distil an explicit requirement list without inventing missing fields."""
    requirements: list[str] = []
    if brief.function.strip():
        requirements.append(f"function: {brief.function.strip()}")
    for technology in brief.technologies:
        if technology.strip():
            requirements.append(f"technology: {technology.strip()}")
    if brief.sector and brief.sector.strip():
        requirements.append(f"sector: {brief.sector.strip()}")
    if brief.scale and brief.scale != "unknown":
        requirements.append(f"scale: {brief.scale}")
    if brief.country and brief.country.strip():
        requirements.append(f"country: {brief.country.strip()}")
    for regulation in brief.regulations:
        if regulation.strip():
            requirements.append(f"regulation: {regulation.strip()}")
    for constraint in brief.constraints:
        if constraint.strip():
            requirements.append(f"constraint: {constraint.strip()}")
    return requirements


def structure_to_components(structure: AgentStructure) -> list[Component]:
    """Flatten module→task hierarchy into searchable components."""
    components: list[Component] = []
    for module_index, module in enumerate(structure.modules):
        for task_index, task in enumerate(module.tasks):
            components.append(
                Component(
                    component_id=stable_component_id(
                        module_index, task_index, task.name
                    ),
                    name=task.name,
                    category=module.name,
                    description=(task.description or "").strip(),
                )
            )
    return ensure_unique_component_ids(components)


def _confidence_from_coverage(budgeted: int, total: int) -> str:
    if total == 0:
        return "low"
    ratio = budgeted / total
    if ratio >= 0.85:
        return "high"
    if ratio >= 0.5:
        return "medium"
    return "low"


def make_graph_nodes(deps: GraphNodeDeps) -> dict[str, Callable[..., Any]]:
    """Build the five node callables closed over the injected dependencies."""

    async def extract_requirements(state: EstimationState) -> dict[str, Any]:
        transcript = state["transcript"]
        estimation_id = _estimation_id()
        with logfire.span(
            "agent.graph.extract_requirements",
            estimation_id=estimation_id,
        ):
            brief = await deps.reformulate(transcript)
            requirements = requirements_from_brief(brief)
            project_brief = brief.model_dump(mode="json")
            logfire.info(
                "extract_requirements_done",
                requirement_count=len(requirements),
                estimation_id=estimation_id,
            )
            log.info(
                "graph_extract_requirements",
                requirement_count=len(requirements),
                estimation_id=estimation_id,
            )
            return {
                "project_brief": project_brief,
                "requirements": requirements,
            }

    async def classify_components(state: EstimationState) -> dict[str, Any]:
        estimation_id = _estimation_id()
        with logfire.span(
            "agent.graph.classify_components",
            estimation_id=estimation_id,
        ):
            brief = EstimationQuery.model_validate(state["project_brief"])
            structure = await deps.propose_structure(brief)
            components = structure_to_components(structure)
            logfire.info(
                "classify_components_done",
                component_count=len(components),
                estimation_id=estimation_id,
            )
            log.info(
                "graph_classify_components",
                component_count=len(components),
                estimation_id=estimation_id,
            )
            return {"components": components}

    async def search_budgets_node(state: EstimationState) -> dict[str, Any]:
        """Search one component at a time — serial baseline for the live session."""
        estimation_id = _estimation_id()
        with logfire.span(
            "agent.graph.search_budgets",
            estimation_id=estimation_id,
            component_count=len(state.get("components") or []),
        ):
            matches: list[BudgetMatch] = []
            brief = EstimationQuery.model_validate(state.get("project_brief") or {})
            # Corpus stores lowercase sectors (e.g. "logistics"); reformulation
            # may emit Title Case — normalise so the hard filter does not wipe recall.
            sectors = [brief.sector.strip().lower()] if brief.sector else None
            for component in state.get("components") or []:
                query = compose_task_search_text(
                    component["category"],
                    component["name"],
                    component.get("description") or None,
                )
                with logfire.span(
                    "agent.graph.search_budgets.component",
                    estimation_id=estimation_id,
                    component_id=component["component_id"],
                ):
                    result = await search_budgets(
                        {
                            "query": query,
                            "filters": {
                                "sectors": sectors,
                                "component_type": component["category"],
                            },
                        },
                        backend=deps.retrieval_backend,
                    )
                for item in result.get("items") or []:
                    hours = item.get("estimated_hours")
                    if hours is None:
                        continue
                    matches.append(
                        BudgetMatch(
                            component_id=component["component_id"],
                            chunk_id=int(item["id"]),
                            reference_budget_id=item.get("budget_id"),
                            amount=float(hours),
                            distance=float(item.get("distance") or 0.0),
                        )
                    )
            logfire.info(
                "search_budgets_done",
                match_count=len(matches),
                estimation_id=estimation_id,
            )
            log.info(
                "graph_search_budgets",
                match_count=len(matches),
                estimation_id=estimation_id,
            )
            return {"budget_matches": matches}

    async def generate_estimate(state: EstimationState) -> dict[str, Any]:
        estimation_id = _estimation_id()
        with logfire.span(
            "agent.graph.generate_estimate",
            estimation_id=estimation_id,
        ):
            components = state.get("components") or []
            by_id: dict[str, list[BudgetMatch]] = defaultdict(list)
            for match in state.get("budget_matches") or []:
                by_id[match["component_id"]].append(match)

            calc_components = []
            for component in components:
                refs = [m["amount"] for m in by_id[component["component_id"]]]
                calc_components.append(
                    {
                        "name": component["component_id"],
                        "reference_amounts": refs,
                    }
                )
            calc = calculate_estimate({"components": calc_components})

            estimate_components: list[AgentComponent] = []
            assumptions: list[str] = []
            budgeted = 0
            for component, row in zip(components, calc["components"], strict=True):
                matches = by_id[component["component_id"]]
                cited = [m["chunk_id"] for m in matches]
                hours = float(row["estimated_hours"])
                if row["unbudgeted"]:
                    assumptions.append(
                        f"No historical references for "
                        f"{component['category']}/{component['name']}; hours set to 0."
                    )
                    rationale = (
                        "No historical references found; left at 0h pending review."
                    )
                else:
                    budgeted += 1
                    rationale = (
                        f"Median of {row['reference_count']} historical hour(s) "
                        f"plus 15% contingency for "
                        f"{component['category']}/{component['name']}."
                    )
                estimate_components.append(
                    AgentComponent(
                        name=component["name"],
                        estimated_hours=hours,
                        cited_chunk_ids=cited,
                        rationale=rationale,
                    )
                )

            draft = AgentEstimate(
                components=estimate_components,
                total_hours=float(calc["total_hours"]),
                assumptions=assumptions,
                confidence=_confidence_from_coverage(budgeted, len(components)),
            )
            logfire.info(
                "generate_estimate_done",
                component_count=len(estimate_components),
                total_hours=draft.total_hours,
                estimation_id=estimation_id,
            )
            return {"estimate": draft.model_dump(mode="json")}

    async def validate_and_consolidate(state: EstimationState) -> dict[str, Any]:
        estimation_id = _estimation_id()
        with logfire.span(
            "agent.graph.validate_and_consolidate",
            estimation_id=estimation_id,
        ):
            estimate = AgentEstimate.model_validate(state["estimate"])
            by_id: dict[str, list[BudgetMatch]] = defaultdict(list)
            for match in state.get("budget_matches") or []:
                by_id[match["component_id"]].append(match)

            components = state.get("components") or []
            validate_rows = []
            for component, estimated_component in zip(
                components, estimate.components, strict=True
            ):
                refs = [m["amount"] for m in by_id[component["component_id"]]]
                validate_rows.append(
                    {
                        "name": component["name"],
                        "estimated_hours": estimated_component.estimated_hours,
                        "reference_amounts": refs,
                    }
                )

            validation = validate_estimate(
                {
                    "components": validate_rows,
                    "total_hours": estimate.total_hours,
                }
            )
            AgentEstimate.model_validate(estimate.model_dump())
            if validation["ok"]:
                update: dict[str, Any] = {"status": "validated"}
            else:
                update = {
                    "status": "needs_review",
                    "errors": list(validation["issues"]),
                }
            logfire.info(
                "validate_and_consolidate_done",
                status=update["status"],
                issue_count=len(validation["issues"]),
                estimation_id=estimation_id,
            )
            log.info(
                "graph_validate_and_consolidate",
                status=update["status"],
                issue_count=len(validation["issues"]),
                estimation_id=estimation_id,
            )
            return update

    return {
        "extract_requirements": extract_requirements,
        "classify_components": classify_components,
        "search_budgets": search_budgets_node,
        "generate_estimate": generate_estimate,
        "validate_and_consolidate": validate_and_consolidate,
    }


# Re-export types used by adapters/tests without creating a circular import.
__all__ = [
    "GraphNodeDeps",
    "LegacyRetrievalBackend",
    "ReformulateFn",
    "StructureFn",
    "make_graph_nodes",
    "requirements_from_brief",
    "structure_to_components",
]
