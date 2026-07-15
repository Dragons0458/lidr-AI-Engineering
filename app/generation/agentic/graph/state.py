"""Typed shared state for the Session 13 estimation graph.

Fields without a reducer overwrite on each update. ``budget_matches`` and
``errors`` accumulate with ``operator.add`` so a future fan-out can fan-in
without losing partial results. Every value must stay JSON-serializable —
no clients, SDK responses, or DB connections belong here.
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


class EstimationState(TypedDict, total=False):
    """Partial updates are merges; only annotated list fields accumulate."""

    transcript: str
    project_brief: dict
    requirements: list[str]
    components: list[Component]
    budget_matches: Annotated[list[BudgetMatch], operator.add]
    estimate: dict
    status: GraphStatus
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
