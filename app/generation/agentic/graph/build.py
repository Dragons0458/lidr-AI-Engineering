"""Compile Session 13 estimation StateGraphs.

``build_sequential_graph`` — the pre-exercise five-node pipeline (tests).
``build_estimation_graph`` — the live multi-agent flow with human gates.
"""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.generation.agentic.graph.agent_nodes import (
    MultiAgentDeps,
    make_multiagent_nodes,
)
from app.generation.agentic.graph.nodes import GraphNodeDeps, make_graph_nodes
from app.generation.agentic.graph.state import EstimationState

log = structlog.get_logger()

NODE_NAMES = (
    "extract_requirements",
    "classify_components",
    "search_budgets",
    "generate_estimate",
    "validate_and_consolidate",
)

MULTIAGENT_NODE_NAMES = (
    "classifier_agent",
    "structure_agent",
    "human_gate_structure",
    "estimate_task_hours",
    "recover_and_handover",
    "analysis_agent",
    "human_gate_analysis",
    "proposal_agent",
)


def build_sequential_graph(
    deps: GraphNodeDeps,
    *,
    checkpointer: Any | None = None,
):
    """Wire the five legacy nodes with direct edges and optionally a checkpointer."""
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


def fan_out_hours(state: EstimationState):
    """Conditional edge after gate 1: one ``Send`` per approved task."""
    modules = state.get("approved_modules") or []
    sends = [
        Send(
            "estimate_task_hours",
            {
                "module": m["name"],
                "task": t["name"],
                "description": t.get("description"),
            },
        )
        for m in modules
        for t in (m.get("tasks") or [])
        if t.get("name")
    ]
    return sends or "recover_and_handover"


def route_after_gate2(state: EstimationState, *, proposal_enabled: bool = True) -> str:
    """Conditional edge after gate 2: optional proposal node or END."""
    decision = state.get("gate2_decision") or {}
    if proposal_enabled and decision.get("want_proposal"):
        return "proposal"
    return "end"


def build_estimation_graph(
    deps: MultiAgentDeps,
    *,
    checkpointer: Any | None = None,
    proposal_enabled: bool = True,
):
    """Wire the multi-agent pipeline with handovers, fan-out and human gates."""
    nodes = make_multiagent_nodes(deps)
    builder = StateGraph(EstimationState)
    for name in MULTIAGENT_NODE_NAMES:
        builder.add_node(name, nodes[name])

    def _route_after_gate2(state: EstimationState) -> str:
        return route_after_gate2(state, proposal_enabled=proposal_enabled)

    builder.add_edge(START, "classifier_agent")
    builder.add_edge("structure_agent", "human_gate_structure")
    builder.add_conditional_edges(
        "human_gate_structure",
        fan_out_hours,
        ["estimate_task_hours", "recover_and_handover"],
    )
    builder.add_edge("estimate_task_hours", "recover_and_handover")
    builder.add_edge("analysis_agent", "human_gate_analysis")
    builder.add_conditional_edges(
        "human_gate_analysis",
        _route_after_gate2,
        {"proposal": "proposal_agent", "end": END},
    )
    builder.add_edge("proposal_agent", END)

    return builder.compile(checkpointer=checkpointer)
