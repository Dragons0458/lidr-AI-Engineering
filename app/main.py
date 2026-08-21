from contextlib import asynccontextmanager
import logging
import time
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import structlog
from slowapi.errors import RateLimitExceeded

from app.api import config as config_api
from app.api.embeddings import router as embeddings_router
from app.api.estimations import router as estimation_router
from app.api.ingestion import router as ingestion_router
from app.api.rate_limiting import limiter, rate_limit_exceeded_handler
from app.api.routers.corpus_index import router as corpus_index_router
from app.api.routers.estimate import router as estimate_router
from app.api.routers.estimate_agent import router as estimate_agent_router
from app.api.routers.estimate_graph import router as estimate_graph_router
from app.api.routers.estimate_stages import router as estimate_stages_router
from app.api.routers.estimate_supervisor import router as estimate_supervisor_router
from app.api.routers.estimate_tasks import router as estimate_tasks_router
from app.api.routers.health import router as health_router
from app.api.routers.observability import router as observability_router
from app.api.routers.retrieval import router as retrieval_router
from app.api.routers.retrieval_advanced import router as retrieval_advanced_router
from app.api.search import router as search_router
from app.api.security import require_service_token
from app.api.sessions import router as sessions_router
from app.config import get_settings
from app.foundation.observability.logfire_setup import (
    configure_logfire,
    instrument_asyncpg,
    instrument_fastapi_app,
    instrument_http_clients,
)
from app.foundation.observability.metrics import (
    metrics_from_usage,
    record_request_metrics,
)
from app.foundation.observability.usage import start_usage
from app.foundation.persistence.langgraph import (
    close_langgraph_runtime,
    open_langgraph_runtime,
)


def configure_logging(env: str, log_level: str) -> None:
    """Configure structlog rendering and level for the selected environment."""
    logging.basicConfig(level=getattr(logging, log_level), force=True)
    renderer = (
        structlog.processors.JSONRenderer()
        if env == "production"
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.EventRenamer("event"),
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.APP_ENV, settings.LOG_LEVEL)
    log = structlog.get_logger()
    instrument_http_clients()
    instrument_asyncpg()
    try:
        from app.dependencies import get_catalog

        catalog = get_catalog()
        log.info(
            "catalog_loaded",
            version=catalog.version,
            sources_total=len(catalog.sources),
            sources_included=len(catalog.included_sources()),
        )
    except Exception as exc:
        log.error("catalog_load_failed", error=str(exc)[:400])

    app.state.graph_runtime = None
    app.state.multiagent_deps = None
    if settings.LANGGRAPH_ENABLED:
        try:
            from app.dependencies import (
                get_async_openai_client,
                get_llm_wrapper,
                get_runtime_retrieval_config,
            )
            from app.domain.graph_estimation import (
                build_default_multiagent_deps,
                compile_estimation_graph,
            )
            from app.domain.supervisor_estimation import (
                build_default_supervisor_deps,
                compile_supervisor_graph,
            )

            runtime_retrieval = get_runtime_retrieval_config()
            multiagent_deps = build_default_multiagent_deps(
                client=get_async_openai_client(),
                settings=settings,
                runtime_retrieval=runtime_retrieval,
            )
            app.state.multiagent_deps = multiagent_deps
            app.state.graph_runtime = await open_langgraph_runtime(
                settings.DATABASE_URL,
                build_graph=lambda checkpointer: compile_estimation_graph(
                    multiagent_deps,
                    checkpointer=checkpointer,
                    proposal_enabled=settings.GRAPH_PROPOSAL_ENABLED,
                ),
            )
            if settings.SUPERVISOR_ENABLED and app.state.graph_runtime is not None:
                try:
                    supervisor_deps = build_default_supervisor_deps(
                        client=get_async_openai_client(),
                        settings=settings,
                        llm_wrapper=get_llm_wrapper(),
                    )
                    app.state.graph_runtime.supervisor_graph = compile_supervisor_graph(
                        supervisor_deps,
                        checkpointer=app.state.graph_runtime.checkpointer,
                    )
                    log.info("supervisor_graph_ready")
                except Exception as supervisor_exc:  # noqa: BLE001
                    app.state.graph_runtime.supervisor_graph = None
                    log.error(
                        "supervisor_graph_startup_failed",
                        error=str(supervisor_exc)[:400],
                        error_type=type(supervisor_exc).__name__,
                    )
        except Exception as exc:  # noqa: BLE001
            app.state.graph_runtime = None
            log.error(
                "langgraph_runtime_startup_failed",
                error=str(exc)[:400],
                error_type=type(exc).__name__,
            )

    log.info("application_started", environment=settings.APP_ENV)
    try:
        yield
    finally:
        await close_langgraph_runtime(getattr(app.state, "graph_runtime", None))
        app.state.graph_runtime = None
        app.state.multiagent_deps = None
        log.info("application_shutdown")


settings = get_settings()

app = FastAPI(
    title="Estimation API",
    description="""
API para generar estimaciones de proyectos de software a partir de transcripciones de reuniones usando LLMs.

### Funcionalidades:
- Generación automática de estimaciones
- Integración con modelos de IA
- Métricas de uso (tokens, coste)

### Endpoints principales:
- POST /api/v1/estimate → Generar estimación
- POST /api/v1/estimate/stream → Generar estimación en streaming
- POST /embeddings/ingest → Ingestar presupuesto en Postgres + pgvector
- POST /search → Búsqueda semántica por distancia coseno (SQL)
- POST /embeddings/compare → Comparar estrategias de chunking (en memoria)
- POST /v1/retrieval/search → Búsqueda filtrada con API key (S09)
- POST /v1/retrieval/advanced-search → Multi-index advanced retrieval (S10)
- POST /v1/estimate/from-transcript → Estimación fundamentada (S09)
- POST /v1/estimate/stages/* → Wizard RAG por etapas (S09/S10)
- POST /v1/estimate/tasks/hours → Per-task hours from historical corpus (S10)
- POST /v1/estimate/agent/structure → Estructura agéntica sin horas (S12)
- POST /v1/estimate/agent/hours → Horas deterministas + recovery agéntico (S12)
- POST /v1/estimate/agent/graph → Grafo multiagente LangGraph con gates humanos (S13)
- POST /v1/estimate/agent/graph/{estimation_id}/resume → Reanudar gate humano (S13)
- GET /v1/estimate/agent/graph/{estimation_id}/state → Estado del grafo (S13)
- POST /v1/estimate/agent/graph/stream → Inicio en background con feed de actividad (S13)
- GET /v1/estimate/agent/graph/{estimation_id}/progress → Progreso en vivo (S13)
- POST /v1/estimate/agent/graph/{estimation_id}/proposal → Propuesta comercial (S13)
- GET /api/v1/config/models → Configuración runtime de modelos
- GET /api/v1/config/retrieval → Configuración runtime de recuperación (S10)
- GET /v1/observability/metrics → Resumen de latencia / coste / errores (S16)
- GET /v1/observability/requests → Filas por petición (join del harness S16)
- GET /health/ready → Readiness (Postgres + Redis)
- POST /sessions → Crear sesión en memoria
- POST /sessions/{session_id}/estimate → Estimar usando sesión y adjuntos
""",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(require_service_token)],
)

configure_logfire(
    token=settings.LOGFIRE_TOKEN,
    service_name=settings.LOGFIRE_SERVICE_NAME,
    environment=settings.APP_ENV,
)
instrument_fastapi_app(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

_METRICS_SKIP_PREFIXES = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/v1/observability",
)
_METRICS_PREFIXES = ("/api/v1/estimate", "/v1/estimate")


def _should_record_metrics(path: str) -> bool:
    """Instrument estimate routes only; never health, docs, or the metrics API."""
    if any(
        path == prefix or path.startswith(f"{prefix}/")
        for prefix in _METRICS_SKIP_PREFIXES
    ):
        return False
    if path.startswith("/health"):
        return False
    return any(
        path == prefix or path.startswith(f"{prefix}/") or path.startswith(prefix)
        for prefix in _METRICS_PREFIXES
    )


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Inner metrics middleware — runs after ``request_id`` has bound the id.

    Registered *before* ``request_id_middleware`` so Starlette treats request_id
    as the outer layer (last added = first executed).
    """
    path = request.url.path
    if not _should_record_metrics(path) or not get_settings().METRICS_ENABLED:
        return await call_next(request)

    usage = start_usage()
    started = time.perf_counter()
    status_code = 500
    error_type = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        error_type = type(exc).__name__
        raise
    finally:
        latency_ms = (time.perf_counter() - started) * 1000
        request_id = getattr(request.state, "request_id", None) or request.headers.get(
            "X-Request-ID", "unknown"
        )
        metric = metrics_from_usage(
            request_id=str(request_id),
            route=path,
            http_status=status_code,
            latency_ms=latency_ms,
            usage=usage,
            error_type=error_type,
        )
        record_request_metrics(metric)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.unbind_contextvars("request_id")
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(health_router)
app.include_router(estimation_router, prefix="/api/v1")
app.include_router(sessions_router, prefix="/api/v1")
app.include_router(ingestion_router)
app.include_router(embeddings_router)
app.include_router(search_router)
app.include_router(config_api.router)
app.include_router(retrieval_router)
app.include_router(retrieval_advanced_router)
app.include_router(estimate_router)
app.include_router(estimate_stages_router)
app.include_router(estimate_tasks_router)
app.include_router(estimate_agent_router)
app.include_router(estimate_graph_router)
app.include_router(estimate_supervisor_router)
app.include_router(corpus_index_router)
app.include_router(observability_router)
