# =============================================================================
# Session 15 — multi-target image
# Shared expensive layers (uv sync + torch/sentence-transformers + spaCy);
# distinct runtime targets for ai-service (FastAPI) and web (Streamlit).
# No secrets as ARG/ENV — injected at runtime via compose env_file.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage: builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-install-project --no-dev

# Spanish spaCy model for Presidio (Session 6). Must run here: runtime has no uv/pip.
RUN uv run python -m spacy download es_core_news_md


# -----------------------------------------------------------------------------
# Stage: base (common runtime)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS base

RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --create-home appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1


# -----------------------------------------------------------------------------
# Stage: ai-service (FastAPI — private, no host ports in prod-like compose)
# -----------------------------------------------------------------------------
FROM base AS ai-service

COPY app/ /app/app/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/alembic.ini
COPY data/ /app/data/
COPY scripts/ /app/scripts/

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# -----------------------------------------------------------------------------
# Stage: web (Streamlit — the only public surface)
# Copies app/ as well: streamlit_ui imports domain schemas and LLM helpers
# (ADR-003 debt: network boundary is intact; code boundary is not).
# -----------------------------------------------------------------------------
FROM base AS web

COPY app/ /app/app/
COPY streamlit_ui/ /app/streamlit_ui/
# Session 15 check 4: smoke must run FROM web (public → private path).
COPY scripts/smoke_test_s15.py /app/scripts/smoke_test_s15.py
COPY scripts/data/s15_smoke_transcript.txt /app/scripts/data/s15_smoke_transcript.txt

RUN mkdir -p /app/streamlit_ui/data && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"]

CMD ["streamlit", "run", "streamlit_ui/home.py", "--server.address=0.0.0.0", "--server.port=8501"]
