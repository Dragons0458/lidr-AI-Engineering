from contextlib import asynccontextmanager
from datetime import datetime
import logging
from uuid import uuid4

from fastapi import FastAPI, Request
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
from app.api.routers.estimate_stages import router as estimate_stages_router
from app.api.routers.estimate_tasks import router as estimate_tasks_router
from app.api.routers.retrieval import router as retrieval_router
from app.api.routers.retrieval_advanced import router as retrieval_advanced_router
from app.api.search import router as search_router
from app.api.sessions import router as sessions_router
from app.config import get_settings


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
    log.info("application_started", environment=settings.APP_ENV)
    yield
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
- GET /api/v1/config/models → Configuración runtime de modelos
- GET /api/v1/config/retrieval → Configuración runtime de recuperación (S10)
- GET /health → Estado del servicio
- POST /sessions → Crear sesión en memoria
- POST /sessions/{session_id}/estimate → Estimar usando sesión y adjuntos
""",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


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
app.include_router(corpus_index_router)


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}
