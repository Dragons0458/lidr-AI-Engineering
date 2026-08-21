# Scalability audit (not implemented in Session 15)

File-and-line notes against **this** repo. Nothing here is part of S15
delivery; it is the order of work if the single-VM kit ever has to take
concurrent load.

## Concurrency ceiling

There is no Puma. The public layer is Streamlit (one process, one thread
per browser session). The AI service is a single uvicorn worker.

Synchronous LLM calls are wrapped in `asyncio.to_thread`, which uses the
default executor: `min(32, os.cpu_count() + 4)`. On a `t3.medium`
(2 vCPU) that is **6 threads for the whole process**.

`asyncio.to_thread` occurrences in `app/` (27):

| File | Count | Typical work |
| --- | --- | --- |
| `app/generation/rag/estimator.py` | 6 | generation + idempotency + embed |
| `app/domain/graph_estimation.py` | 3 | structured LLM |
| `app/domain/supervisor_estimation.py` | 3 | structured LLM |
| `app/generation/rag/query_reformulator.py` | 2 | structured LLM |
| `app/generation/rag/retrieval/advanced_pipeline.py` | 2 | embed + rerank |
| Others (1 each) | 11 | embed, rerank, retrieve |

Two overlapping graph estimates saturate the executor; a third waits.
The cheap lever is the CAG hit rate (`app/generation/cag/`), not another
vCPU.

## The probe that punishes instead of measuring

`GET /health/ready` (`app/api/routers/health.py`) uses the **sync** engine
(`create_engine_from_settings`) and `redis.from_url` inside an `async`
endpoint. Each probe blocks the event loop. Compose HEALTHCHECK is every
30s; a long estimate in flight plus a blocked loop is exactly the pattern
that restarts a container that was only busy.

Derived action (not S15): move the probe onto async clients
(`create_async_engine_from_settings`, `redis.asyncio`).

## What breaks on a second `ai-service` replica

| Risk | Where | What happens |
| --- | --- | --- |
| In-memory conversation sessions | `app/generation/conversation/store.py` (`Session._sessions`) | Sticky-less replica → empty session / 404 |
| Rate limiter has no `storage_uri` | `app/api/rate_limiting.py:22` (`Limiter(key_func=…)`) | Per-process counters; limits silently dilute |
| `BackgroundTasks` of the process | `app/api/routers/estimate_graph.py` (stream start ~181) | Progress lives on the replica that accepted the POST |
| Idempotency / activity log degrade on Redis miss | `app/generation/rag/idempotency.py:27`, `app/generation/agentic/graph/activity.py:159` | In-process dict fallback → duplicates and empty feeds |
| Postgres pools unconfigured | `app/foundation/persistence/database.py:23`, `async_database.py:29`, LangGraph `AsyncConnectionPool` `min_size=1 max_size=5` (`langgraph.py:52`) | Default SQLAlchemy pool × N replicas |
| `migrate` runs `alembic upgrade head` on every boot | `docker-compose.yml` service `migrate` | Two replicas racing migrations |

## The real ceiling is provider quota

CPU is not the bill. OpenAI/Anthropic tokens are. CAG exact + semantic
caches already exist; **hit rate** is the cost metric. Measure p95/p99,
never the mean. Saturate the executor (queue depth) before looking at
CPU utilisation.

Correlation already in place: `X-Request-ID` middleware in `app/main.py`
(~line 221). Export traces with `LOGFIRE_TOKEN` (`app/config.py`,
`LOGFIRE_SERVICE_NAME`).

## Suggested order if this ever has to be fixed

1. CAG observability (hit rate by bucket) — biggest cost win, smallest code.
2. Async `/health/ready` — stop killing busy workers.
3. Redis-backed rate limiter + drop in-memory session store.
4. Stop running `migrate` as a sidecar of every replica (CI/CD migrate job).
5. Only then: second uvicorn worker / second replica.

None of the above is Session 15.
