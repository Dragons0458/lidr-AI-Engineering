from datetime import datetime

from fastapi import FastAPI

from app.routers.estimations import router as estimation_router
from app.routers.sessions import router as sessions_router

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
)

app.include_router(estimation_router, prefix="/api/v1")
app.include_router(sessions_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}
