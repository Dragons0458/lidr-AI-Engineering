"""Typed shared state for the Session 13 estimation graph.

Fields without a reducer overwrite on each update. ``budget_matches``,
``errors`` and ``task_hours`` use reducers so fan-out / resume stay idempotent.
Every value must stay JSON-serializable — no clients, SDK responses, or DB
connections belong here.
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Literal, TypedDict

GraphStatus = Literal["validated", "needs_review"]

_ID_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


class Component(TypedDict):
    """One searchable work unit (a task); ``category`` is the module name."""

    component_id: str
    name: str
    category: str
    description: str


class BudgetMatch(TypedDict):
    """One historical reference tied to a searchable component."""

    component_id: str
    chunk_id: int
    reference_budget_id: str | None
    amount: float
    distance: float


def merge_task_hours(existing: list[dict] | None, new: list[dict] | None) -> list[dict]:
    """Keyed reducer for per-task hours fan-out — dedupe by (module, task)."""
    by_key: dict[tuple[str, str], dict] = {
        (t.get("module"), t.get("task")): t for t in (existing or [])
    }
    for t in new or []:
        by_key[(t.get("module"), t.get("task"))] = t
    return list(by_key.values())


class EstimationState(TypedDict, total=False):
    """Partial updates are merges; annotated list fields accumulate."""

    transcript: str
    estimation_id: str
    project_brief: dict

    # --- classifier_agent -------------------------------------------------- #
    complexity: str | None
    reformulated_transcript: str | None

    # --- legacy sequential pipeline (pre-exercise nodes/tests) ----------- #
    requirements: list[str]
    components: list[Component]
    budget_matches: Annotated[list[BudgetMatch], operator.add]

    # --- structure_agent + human gate 1 ------------------------------------ #
    structure: dict | None
    approved_modules: list[dict] | None
    gate1_decision: dict | None

    # --- hours fan-out + merged estimate ----------------------------------- #
    task_hours: Annotated[list[dict], merge_task_hours]
    estimate: dict | None

    # --- analysis_agent + human gate 2 ------------------------------------- #
    analysis_report: dict | None
    gate2_decision: dict | None

    # --- proposal_agent (bonus) -------------------------------------------- #
    proposal: str | None

    status: GraphStatus | None
    errors: Annotated[list[str], operator.add]


def stable_component_id(module_index: int, task_index: int, name: str) -> str:
    """Build a stable id from position; name is only a readable suffix."""
    slug = _ID_SAFE.sub("-", name.strip().lower()).strip("-")[:40] or "task"
    return f"m{module_index}-t{task_index}-{slug}"


def ensure_unique_component_ids(components: list[Component]) -> list[Component]:
    """Detect collisions and suffix duplicates so joins stay unambiguous."""
    seen: dict[str, int] = {}
    unique: list[Component] = []
    for component in components:
        base = component["component_id"]
        count = seen.get(base, 0)
        seen[base] = count + 1
        if count == 0:
            unique.append(component)
            continue
        unique.append({**component, "component_id": f"{base}~{count}"})
    return unique


def assert_serializable_state(state: dict) -> None:
    """Raise if any value is not a JSON-friendly scalar/container."""
    import json

    json.dumps(state)
