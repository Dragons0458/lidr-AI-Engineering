"""Compile the Session 13 sequential estimation StateGraph.

Topology (Level 1 — no conditionals, no cycles, no fan-out):

    START → extract_requirements → classify_components → search_budgets
          → generate_estimate → validate_and_consolidate → END
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.generation.agentic.graph.nodes import GraphNodeDeps, make_graph_nodes
from app.generation.agentic.graph.state import EstimationState

NODE_NAMES = (
    "extract_requirements",
    "classify_components",
    "search_budgets",
    "generate_estimate",
    "validate_and_consolidate",
)


def build_estimation_graph(
    deps: GraphNodeDeps,
    *,
    checkpointer: Any | None = None,
):
    """Wire the five nodes with direct edges and optionally a checkpointer."""
    nodes = make_graph_nodes(deps)
    builder = StateGraph(EstimationState)
    for name in NODE_NAMES:
        builder.add_node(name, nodes[name])

    builder.add_edge(START, "extract_requirements")
    builder.add_edge("extract_requirements", "classify_components")
    builder.add_edge("classify_components", "search_budgets")
    builder.add_edge("search_budgets", "generate_estimate")
    builder.add_edge("generate_estimate", "validate_and_consolidate")
    builder.add_edge("validate_and_consolidate", END)

    return builder.compile(checkpointer=checkpointer)
