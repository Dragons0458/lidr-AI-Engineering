"""Domain conductor for the Session 13 LangGraph estimation workflow."""

from __future__ import annotations

from typing import Any

import structlog

from app.domain.schemas.graph_estimation import GraphEstimationResponse
from app.foundation.persistence.langgraph import LangGraphRuntime
from app.generation.agentic.agent_loop import run_structure_agent
from app.generation.agentic.agent_schemas import AgentEstimate, AgentStructure
from app.generation.agentic.agent_tools import adapt_recovery_backend
from app.generation.agentic.graph.build import build_estimation_graph
from app.generation.agentic.graph.nodes import GraphNodeDeps
from app.generation.rag.agent_retrieval import make_retrieval_backend
from app.generation.rag.prompt_builder import build_structure_user_message
from app.generation.rag.query_reformulator import reformulate_query
from app.generation.rag.schemas import EstimationQuery

log = structlog.get_logger()


class GraphRuntimeUnavailableError(RuntimeError):
    """Raised when the LangGraph runtime was not started (e.g. no Postgres)."""


class GraphEstimationError(RuntimeError):
    """Raised when the graph run fails for a technical reason."""


def build_default_graph_deps(
    *,
    client: Any,
    model: str,
    reasoning_effort: str,
    top_k: int,
    distance_threshold: float,
    search_mode: str,
    rerank: bool,
) -> GraphNodeDeps:
    """Wire production adapters around existing S9–S12 capabilities."""

    async def reformulate(transcript: str) -> EstimationQuery:
        return await reformulate_query(transcript)

    async def propose_structure(brief: EstimationQuery) -> AgentStructure:
        user_message = build_structure_user_message(brief)
        structure, _trace = await run_structure_agent(
            user_message,
            client=client,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        return structure

    recovery = make_retrieval_backend(
        top_k=top_k,
        distance_threshold=distance_threshold,
        search_mode=search_mode,
        rerank=rerank,
    )
    return GraphNodeDeps(
        reformulate=reformulate,
        propose_structure=propose_structure,
        retrieval_backend=adapt_recovery_backend(recovery),
    )


def compile_estimation_graph(deps: GraphNodeDeps, *, checkpointer: Any):
    """Factory used by the lifespan to compile the production graph once."""
    return build_estimation_graph(deps, checkpointer=checkpointer)


async def run_graph_estimation(
    *,
    estimation_id: str,
    transcript: str,
    runtime: LangGraphRuntime | None,
) -> GraphEstimationResponse:
    """Invoke the compiled graph and map state to the public HTTP contract.

    Only ``transcript`` is sent as input so reusing a ``thread_id`` cannot
    re-append accumulator fields that the checkpointer already holds.
    """
    if runtime is None:
        raise GraphRuntimeUnavailableError(
            "LangGraph estimation runtime is not available."
        )
    config = {"configurable": {"thread_id": estimation_id}}
    try:
        result = await runtime.graph.ainvoke(
            {"transcript": transcript},
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        log.error(
            "graph_estimation_failed",
            estimation_id=estimation_id,
            error_type=type(exc).__name__,
        )
        raise GraphEstimationError("Graph estimation failed.") from exc

    estimate_raw = result.get("estimate")
    status = result.get("status")
    if not estimate_raw or status not in {"validated", "needs_review"}:
        raise GraphEstimationError("Graph estimation produced an incomplete result.")
    estimate = AgentEstimate.model_validate(estimate_raw)
    return GraphEstimationResponse(estimate=estimate, status=status)
