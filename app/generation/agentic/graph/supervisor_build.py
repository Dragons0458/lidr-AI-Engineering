"""Compile the Session 14 supervisor star topology.

Dynamic edges (``supervisor → agent``) are drawn at runtime by ``Command(goto=…)``.
Static edges: ``START → supervisor``, each agent → supervisor, gate → END.
"""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph

from app.generation.agentic.graph.supervisor_nodes import (
    SupervisorDeps,
    make_supervisor_nodes,
)
from app.generation.agentic.graph.supervisor_state import SupervisorState

log = structlog.get_logger()

AGENT_NODE_NAMES = (
    "requirements_extractor",
    "budget_searcher",
    "estimate_generator",
    "coherence_validator",
)


def build_supervisor_graph(
    deps: SupervisorDeps,
    *,
    checkpointer: Any = None,
):
    """Build and compile the supervisor graph closed over ``deps``."""
    nodes = make_supervisor_nodes(deps)
    builder = StateGraph(SupervisorState)

    builder.add_node(
        "supervisor",
        nodes["supervisor"],
        destinations=(*AGENT_NODE_NAMES, "human_review_gate"),
    )
    for name in AGENT_NODE_NAMES:
        builder.add_node(name, nodes[name])
    builder.add_node("human_review_gate", nodes["human_review_gate"])

    builder.add_edge(START, "supervisor")
    for name in AGENT_NODE_NAMES:
        builder.add_edge(name, "supervisor")
    builder.add_edge("human_review_gate", END)

    log.info("supervisor_graph_compiled", agents=list(AGENT_NODE_NAMES))
    return builder.compile(checkpointer=checkpointer)
