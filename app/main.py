from contextlib import asynccontextmanager
from datetime import datetime
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.config import get_settings
from app.routers.estimations import router as estimation_router
from app.routers.sessions import router as sessions_router


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


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}
