"""LangGraph AsyncPostgresSaver lifecycle over the existing DATABASE_URL.

The checkpointer is short-term execution memory for one ``thread_id``. It does
not replace business persistence. The pool is opened once in the FastAPI
lifespan and closed on shutdown — never per request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

log = structlog.get_logger()


@dataclass
class LangGraphRuntime:
    """Compiled graph plus the resources it needs to survive requests."""

    pool: AsyncConnectionPool
    checkpointer: AsyncPostgresSaver
    graph: Any
    # Session 14 supervisor star graph — shares the same checkpointer/pool.
    supervisor_graph: Any | None = None


def to_libpq_conninfo(database_url: str) -> str:
    """Derive a libpq/psycopg URL from the SQLAlchemy ``DATABASE_URL``."""
    raw = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    raw = raw.replace("postgresql+asyncpg://", "postgresql://", 1)
    parsed = urlparse(raw)
    if parsed.scheme not in {"postgresql", "postgres"}:
        raise ValueError(
            f"Unsupported database URL scheme for LangGraph: {parsed.scheme!r}"
        )
    return urlunparse(parsed)


async def open_langgraph_runtime(
    database_url: str,
    *,
    build_graph,
) -> LangGraphRuntime:
    """Open the pool, set up checkpoint tables, and compile the graph."""
    conninfo = to_libpq_conninfo(database_url)
    pool = AsyncConnectionPool(
        conninfo=conninfo,
        min_size=1,
        max_size=5,
        timeout=5.0,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "connect_timeout": 5,
        },
        open=False,
    )
    await pool.open()
    checkpointer = AsyncPostgresSaver(conn=pool)
    try:
        await checkpointer.setup()
        graph = build_graph(checkpointer=checkpointer)
    except Exception:
        await pool.close()
        raise
    log.info("langgraph_runtime_ready")
    return LangGraphRuntime(pool=pool, checkpointer=checkpointer, graph=graph)


async def close_langgraph_runtime(runtime: LangGraphRuntime | None) -> None:
    """Close the connection pool; safe to call with ``None``."""
    if runtime is None:
        return
    try:
        await runtime.pool.close()
        log.info("langgraph_runtime_closed")
    except Exception as exc:  # noqa: BLE001
        log.warning("langgraph_runtime_close_failed", error=str(exc)[:200])
