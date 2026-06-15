from contextlib import asynccontextmanager
from datetime import datetime
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.api import config as config_api
from app.api.embeddings import router as embeddings_router
from app.api.estimations import router as estimation_router
from app.api.ingestion import router as ingestion_router
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
- GET /api/v1/config/models → Configuración runtime de modelos
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

app.include_router(estimation_router, prefix="/api/v1")
app.include_router(sessions_router, prefix="/api/v1")
app.include_router(ingestion_router)
app.include_router(embeddings_router)
app.include_router(search_router)
app.include_router(config_api.router)


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}
