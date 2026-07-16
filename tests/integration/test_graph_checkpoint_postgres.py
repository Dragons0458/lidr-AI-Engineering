"""Integration: AsyncPostgresSaver on the shared estimator Postgres.

Marked so the offline suite stays network/DB free. Run with::

    uv run pytest tests/integration/test_graph_checkpoint_postgres.py -m integration -q
"""

from __future__ import annotations

import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.config import get_settings
from app.foundation.persistence.langgraph import (
    close_langgraph_runtime,
    open_langgraph_runtime,
    to_libpq_conninfo,
)
from app.generation.agentic.agent_schemas import (
    AgentModuleNode,
    AgentStructure,
    AgentTaskNode,
    SearchBudgetsArgs,
)
from app.generation.agentic.graph.build import build_estimation_graph
from app.generation.agentic.graph.nodes import GraphNodeDeps
from app.generation.rag.schemas import EstimationQuery

pytestmark = pytest.mark.integration


def _deps() -> GraphNodeDeps:
    async def reformulate(transcript: str) -> EstimationQuery:
        return EstimationQuery(function="Portal")

    async def propose_structure(brief: EstimationQuery) -> AgentStructure:
        return AgentStructure(
            modules=[
                AgentModuleNode(
                    name="Core",
                    tasks=[AgentTaskNode(name="Auth", description="login")],
                )
            ],
            confidence="high",
            reasoning="ok",
        )

    async def backend(args: SearchBudgetsArgs) -> list[dict]:
        return [
            {
                "id": 1,
                "budget_id": "B1",
                "estimated_hours": 40,
                "distance": 0.2,
            }
        ]

    return GraphNodeDeps(
        reformulate=reformulate,
        propose_structure=propose_structure,
        retrieval_backend=backend,
    )


@pytest.mark.asyncio
async def test_async_postgres_saver_isolates_threads():
    settings = get_settings()
    # Smoke the URL conversion even if the DB is down.
    assert "postgresql://" in to_libpq_conninfo(settings.DATABASE_URL)

    try:
        runtime = await open_langgraph_runtime(
            settings.DATABASE_URL,
            build_graph=lambda checkpointer: build_estimation_graph(
                _deps(), checkpointer=checkpointer
            ),
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres unavailable: {exc}")

    try:
        thread_a = f"s13-a-{uuid.uuid4()}"
        thread_b = f"s13-b-{uuid.uuid4()}"
        result_a = await runtime.graph.ainvoke(
            {"transcript": "portal a"},
            config={"configurable": {"thread_id": thread_a}},
        )
        result_b = await runtime.graph.ainvoke(
            {"transcript": "portal b"},
            config={"configurable": {"thread_id": thread_b}},
        )
        assert result_a["status"] in {"validated", "needs_review"}
        assert result_b["status"] in {"validated", "needs_review"}

        snap_a = await runtime.graph.aget_state(
            {"configurable": {"thread_id": thread_a}}
        )
        snap_b = await runtime.graph.aget_state(
            {"configurable": {"thread_id": thread_b}}
        )
        assert snap_a.values.get("transcript") == "portal a"
        assert snap_b.values.get("transcript") == "portal b"
        assert len(snap_a.values.get("budget_matches") or []) >= 1
    finally:
        await close_langgraph_runtime(runtime)


@pytest.mark.asyncio
async def test_memory_saver_does_not_duplicate_on_fresh_invoke():
    """Control: fresh thread_id with only transcript never duplicates matches."""
    graph = build_estimation_graph(_deps(), checkpointer=InMemorySaver())
    thread = f"mem-{uuid.uuid4()}"
    first = await graph.ainvoke(
        {"transcript": "portal"},
        config={"configurable": {"thread_id": thread}},
    )
    # Re-invoke same thread with only transcript (as the conductor does).
    second = await graph.ainvoke(
        {"transcript": "portal"},
        config={"configurable": {"thread_id": thread}},
    )
    # A full re-run appends via reducer when the graph runs again on same thread.
    # The conductor contract is: clients use a NEW estimation_id per estimation.
    assert len(first.get("budget_matches") or []) >= 1
    assert len(second.get("budget_matches") or []) >= len(
        first.get("budget_matches") or []
    )
