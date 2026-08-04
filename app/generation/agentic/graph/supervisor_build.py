"""Compile the Session 14 supervisor star topology.

Dynamic edges (``supervisor → agent``) are drawn at runtime by ``Command(goto=…)``.
Static edges: ``START → supervisor``, each agent → supervisor, gate → END
(or gate → persistence_agent → END when persistence is enabled).
"""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph

from app.generation.agentic.graph.supervisor_nodes import (
    SupervisorDeps,
    make_supervisor_nodes,
)
from app.generation.agentic.graph.supervisor_sandbox import verify_tool_grants
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
    verify_tool_grants()
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

    if deps.persistence_enabled:
        builder.add_node("persistence_agent", nodes["persistence_agent"])
        builder.add_edge("human_review_gate", "persistence_agent")
        builder.add_edge("persistence_agent", END)
    else:
        builder.add_edge("human_review_gate", END)

    log.info(
        "supervisor_graph_compiled",
        agents=list(AGENT_NODE_NAMES),
        persistence=deps.persistence_enabled,
        competition=deps.competition_enabled,
    )
    return builder.compile(checkpointer=checkpointer)
