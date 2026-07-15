"""The agent's tools: flat Responses schemas + Python implementations.

Two required tools, one optional:

* ``search_budgets`` — WRAPS the Session 9/10 retrieval pipeline (``retrieve()``);
  it does NOT reimplement retrieval. Its retrieval backend is injectable so a
  student stub (``exercises/session-12/reference_retrieval.py``) can stand in when
  the DB is not up.
* ``calculate_estimate`` — a deterministic, non-LLM cost function.
* ``validate_estimate`` — optional S4-style guardrails over the final estimate.

Schema shape matters: the Responses API uses a **flat** function schema
(``{"type": "function", "name": ..., "parameters": {...}}``), NOT the Chat
Completions shape that nests everything under a ``"function"`` key. Every schema
is ``strict: true``, which forces: every property listed in ``required`` (model
optionality via nullable unions, e.g. ``["object", "null"]``) and
``additionalProperties: false`` at *every* object level.

The tool descriptions are the ONLY thing the model reads to decide when to use a
tool — they are written for a model that never sees this code. Optimising them is
the live-session exercise.
"""

from __future__ import annotations

import statistics
from typing import Any, Awaitable, Callable

import structlog

from app.generation.agentic.agent_schemas import (
    AgentTaskDerivation,
    CalculateEstimateArgs,
    SearchBudgetsArgs,
    ValidateEstimateArgs,
)

log = structlog.get_logger()

# Contingency buffer applied to every component's central estimate. A flat,
# transparent number keeps the cost model auditable — no hidden magic.
CONTINGENCY_FACTOR = 0.15

CONTENT_PREVIEW_CHARS = 160


# --------------------------------------------------------------------------- #
# Flat Responses tool schemas                                                 #
# --------------------------------------------------------------------------- #
SEARCH_BUDGETS_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "search_budgets",
    "description": (
        "Search historical project budgets for work analogous to ONE component or "
        "requirement, and return the matching items with their recorded effort in "
        "engineer-hours. Call this once per component you need to cost (e.g. once "
        "for the payments backend, once for the mobile app). Use a focused, "
        "component-specific query — not the whole project. Returns a list of "
        "historical items, each with an id, a text preview, its sector and its "
        "recorded engineer-hours; use those hours as the reference_amounts for "
        "calculate_estimate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language description of the single component to find "
                    "budgets for, e.g. 'OAuth2 authentication backend with JWT and "
                    "multi-tenant token isolation'."
                ),
            },
            "filters": {
                "type": ["object", "null"],
                "description": "Optional structural filters. Pass null to search across everything.",
                "properties": {
                    "sectors": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Restrict to these client sectors, e.g. ['logistics'].",
                    },
                    "component_type": {
                        "type": ["string", "null"],
                        "description": "Free-text hint about the kind of component, e.g. 'mobile app'.",
                    },
                },
                "required": ["sectors", "component_type"],
                "additionalProperties": False,
            },
        },
        "required": ["query", "filters"],
        "additionalProperties": False,
    },
    "strict": True,
}

CALCULATE_ESTIMATE_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "calculate_estimate",
    "description": (
        "Deterministically compute an effort estimate in engineer-hours from a set "
        "of components and their historical reference amounts. For each component it "
        "takes the median of the reference_amounts and adds a fixed contingency "
        "buffer; it then sums the components into a total. Call this once you have "
        "gathered reference amounts (from search_budgets) for every component. This "
        "does NOT call a model — it is pure arithmetic, so its output is reproducible."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "components": {
                "type": "array",
                "description": "The components to cost.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Component name."},
                        "reference_amounts": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": (
                                "Historical engineer-hours for analogous work, taken "
                                "from search_budgets results. May be empty if nothing "
                                "was found (the component is then flagged)."
                            ),
                        },
                    },
                    "required": ["name", "reference_amounts"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["components"],
        "additionalProperties": False,
    },
    "strict": True,
}

VALIDATE_ESTIMATE_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "validate_estimate",
    "description": (
        "Run sanity-check guardrails over a finished estimate before returning it. "
        "It flags components with no historical reference, components whose hours are "
        "far outside the range of their references, a total that does not match the "
        "sum of the components, and non-positive or implausibly large totals. Call "
        "this as the LAST step, once you have a full estimate, and address any issues "
        "it reports before giving your final answer. Returns {ok, issues}."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "components": {
                "type": "array",
                "description": "The estimate's components, with their final hours and references.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "estimated_hours": {"type": "number"},
                        "reference_amounts": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                    },
                    "required": ["name", "estimated_hours", "reference_amounts"],
                    "additionalProperties": False,
                },
            },
            "total_hours": {
                "type": "number",
                "description": "The estimate's grand total in engineer-hours.",
            },
        },
        "required": ["components", "total_hours"],
        "additionalProperties": False,
    },
    "strict": True,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    SEARCH_BUDGETS_TOOL,
    CALCULATE_ESTIMATE_TOOL,
    VALIDATE_ESTIMATE_TOOL,
]

RECOVERY_SEARCH_BUDGETS_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "search_budgets",
    "description": "Search historical tasks using one focused query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "sectors": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
        },
        "required": ["query", "sectors"],
        "additionalProperties": False,
    },
    "strict": True,
}

DERIVE_TASK_HOURS_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "derive_task_hours",
    "description": (
        "Derive hours for one flagged task from retrieved historical neighbors. "
        "This is the only tool allowed to produce recovered hours."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_ref": {"type": "string"},
            "module": {"type": "string"},
            "task": {"type": "string"},
            "neighbors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "integer"},
                        "budget_id": {"type": ["string", "null"]},
                        "estimated_hours": {"type": "integer"},
                        "distance": {"type": "number"},
                    },
                    "required": [
                        "source_id",
                        "budget_id",
                        "estimated_hours",
                        "distance",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["task_ref", "module", "task", "neighbors"],
        "additionalProperties": False,
    },
    "strict": True,
}

HOURS_TOOL_SCHEMAS: list[dict[str, Any]] = [
    RECOVERY_SEARCH_BUDGETS_TOOL,
    DERIVE_TASK_HOURS_TOOL,
    VALIDATE_ESTIMATE_TOOL,
]


# --------------------------------------------------------------------------- #
# Retrieval backend (injectable)                                              #
# --------------------------------------------------------------------------- #
# A backend is an async callable that takes the validated search args and returns
# a list of plain-dict historical items. The default wraps retrieve(); the student
# stub swaps in a canned one so the loop can be debugged without a database.
LegacyRetrievalBackend = Callable[[SearchBudgetsArgs], Awaitable[list[dict[str, Any]]]]
RetrievalBackend = LegacyRetrievalBackend
RecoveryRetrievalBackend = Callable[
    [str, list[str] | None], Awaitable[list[dict[str, Any]]]
]
ConsensusFn = Callable[[list[tuple[int, float]]], tuple[int, float, float]]


async def default_retrieval_backend(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
    """Adapt the default 5-neighbor, 0.45 recovery backend for legacy calls."""
    from app.generation.rag.agent_retrieval import (
        make_default_legacy_recovery_backend,
    )

    sectors = args.filters.sectors if args.filters else None
    return await make_default_legacy_recovery_backend()(args.query, sectors)


# --------------------------------------------------------------------------- #
# Tool implementations                                                        #
# --------------------------------------------------------------------------- #
async def search_budgets(
    raw_args: dict[str, Any], *, backend: LegacyRetrievalBackend
) -> dict[str, Any]:
    """Retrieve historical budget items for one component."""
    args = SearchBudgetsArgs.model_validate(raw_args)
    items = await backend(args)
    hours = [
        it["estimated_hours"] for it in items if it.get("estimated_hours") is not None
    ]
    summary = (
        f"{len(items)} historical items for {args.query!r}; hours={hours}"
        if items
        else f"no historical items for {args.query!r}"
    )
    log.info("agent_tool_search_budgets", query=args.query, results=len(items))
    return {"items": items, "count": len(items), "summary": summary}


async def recovery_search_budgets(
    raw_args: dict[str, Any], *, backend: RecoveryRetrievalBackend
) -> dict[str, Any]:
    """Adapt the recovery query/sectors contract to its retrieval backend."""
    query = raw_args.get("query")
    sectors = raw_args.get("sectors")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if sectors is not None and (
        not isinstance(sectors, list)
        or any(not isinstance(item, str) for item in sectors)
    ):
        raise ValueError("sectors must be an array of strings or null")
    items = await backend(query.strip(), sectors)
    return {
        "items": items,
        "count": len(items),
        "summary": f"{len(items)} historical task matches",
    }


def derive_task_hours(
    raw_args: dict[str, Any], *, consensus: ConsensusFn
) -> dict[str, Any]:
    """Derive hours exclusively through the injected deterministic consensus."""
    base = {
        "task_ref": raw_args.get("task_ref"),
        "module": raw_args.get("module"),
        "task": raw_args.get("task"),
        "neighbors": raw_args.get("neighbors", []),
        "has_match": False,
        "estimated_hours": None,
        "reliability": None,
        "dispersion": None,
    }
    derivation = AgentTaskDerivation.model_validate(base)
    if not derivation.neighbors:
        return derivation.model_dump()
    hours, reliability, dispersion = consensus(
        [(item.estimated_hours, item.distance) for item in derivation.neighbors]
    )
    return AgentTaskDerivation(
        task_ref=derivation.task_ref,
        module=derivation.module,
        task=derivation.task,
        estimated_hours=hours,
        reliability=reliability,
        dispersion=dispersion,
        has_match=True,
        neighbors=derivation.neighbors,
    ).model_dump()


def calculate_estimate(raw_args: dict[str, Any]) -> dict[str, Any]:
    """Deterministically cost the components. No LLM."""
    args = CalculateEstimateArgs.model_validate(raw_args)
    breakdown: list[dict[str, Any]] = []
    total = 0.0
    for component in args.components:
        refs = component.reference_amounts
        if refs:
            central = statistics.median(refs)
            hours = round(central * (1 + CONTINGENCY_FACTOR), 1)
            flagged = False
        else:
            # No reference to anchor on: cost nothing and flag it, rather than
            # inventing a number. The agent should notice and search again.
            hours = 0.0
            flagged = True
        total += hours
        breakdown.append(
            {
                "name": component.name,
                "reference_count": len(refs),
                "estimated_hours": hours,
                "unbudgeted": flagged,
            }
        )
    total = round(total, 1)
    log.info(
        "agent_tool_calculate_estimate", components=len(breakdown), total_hours=total
    )
    return {
        "components": breakdown,
        "total_hours": total,
        "contingency_factor": CONTINGENCY_FACTOR,
        "summary": f"total={total}h across {len(breakdown)} components",
    }


def validate_estimate(raw_args: dict[str, Any]) -> dict[str, Any]:
    """S4-style guardrails over the final estimate. No LLM."""
    args = ValidateEstimateArgs.model_validate(raw_args)
    issues: list[str] = []

    component_sum = 0.0
    for component in args.components:
        component_sum += component.estimated_hours
        if not component.reference_amounts:
            issues.append(
                f"{component.name!r} has no historical reference (unbudgeted)."
            )
            continue
        low = min(component.reference_amounts) * 0.5
        high = max(component.reference_amounts) * 2.0
        if not (low <= component.estimated_hours <= high):
            issues.append(
                f"{component.name!r} estimate {component.estimated_hours}h is outside the "
                f"plausible range [{round(low, 1)}, {round(high, 1)}]h implied by its references."
            )

    if args.total_hours <= 0:
        issues.append("Total hours is non-positive.")
    if abs(component_sum - args.total_hours) > 0.5:
        issues.append(
            f"Total {args.total_hours}h does not match the sum of components "
            f"({round(component_sum, 1)}h)."
        )
    # A single-project estimate above ~10 person-years is almost certainly wrong.
    if args.total_hours > 20_000:
        issues.append(
            f"Total {args.total_hours}h is implausibly large for one project."
        )

    ok = not issues
    log.info("agent_tool_validate_estimate", ok=ok, issues=len(issues))
    return {
        "ok": ok,
        "issues": issues,
        "summary": "estimate passed all guardrails"
        if ok
        else f"{len(issues)} issue(s) found",
    }


# --------------------------------------------------------------------------- #
# Dispatch                                                                     #
# --------------------------------------------------------------------------- #
async def dispatch_tool(
    name: str, raw_args: dict[str, Any], *, backend: RetrievalBackend
) -> dict[str, Any]:
    """Route a tool call to its implementation.

    Raises for an unknown tool name; the loop maps any raised exception (including
    ``pydantic.ValidationError`` from bad arguments) to an error string it returns
    to the model, so a bad call never kills the loop.
    """
    if name == "search_budgets":
        return await search_budgets(raw_args, backend=backend)
    if name == "calculate_estimate":
        return calculate_estimate(raw_args)
    if name == "validate_estimate":
        return validate_estimate(raw_args)
    raise ValueError(f"Unknown tool: {name!r}")


async def dispatch_recovery_tool(
    name: str,
    raw_args: dict[str, Any],
    *,
    backend: RecoveryRetrievalBackend,
    consensus: ConsensusFn,
) -> dict[str, Any]:
    """Dispatch a recovery tool without crossing legacy callable signatures."""
    if name == "search_budgets":
        return await recovery_search_budgets(raw_args, backend=backend)
    if name == "derive_task_hours":
        return derive_task_hours(raw_args, consensus=consensus)
    if name == "validate_estimate":
        return validate_estimate(raw_args)
    raise ValueError(f"Unknown recovery tool: {name!r}")


def adapt_recovery_backend(
    backend: RecoveryRetrievalBackend,
) -> LegacyRetrievalBackend:
    """Explicitly adapt a recovery backend for the legacy search contract."""

    async def legacy(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
        sectors = args.filters.sectors if args.filters else None
        return await backend(args.query, sectors)

    return legacy


def adapt_legacy_backend(
    backend: LegacyRetrievalBackend,
) -> RecoveryRetrievalBackend:
    """Explicitly adapt a legacy backend for recovery query/sectors calls."""

    async def recovery(query: str, sectors: list[str] | None) -> list[dict[str, Any]]:
        return await backend(
            SearchBudgetsArgs.model_validate(
                {
                    "query": query,
                    "filters": {
                        "sectors": sectors,
                        "component_type": None,
                    },
                }
            )
        )

    return recovery
