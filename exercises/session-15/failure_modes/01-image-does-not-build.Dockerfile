# Session 15 failure-mode fixture 1 — the image does not build (or builds forever).
#
# Symptom: `/app/uv.lock: not found` OR a 15-minute rebuild on every Python edit.
# Cause: `COPY . .` before `uv sync`, and a single stage with no `target:`.
# Every code change invalidates the dependency layer (torch + sentence-transformers).
#
# Compare with the real `Dockerfile`: `uv sync` on `pyproject.toml`/`uv.lock` first,
# then `COPY app/`, with named targets `builder` / `base` / `ai-service` / `web`.

FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY . .

RUN uv sync --frozen --no-install-project --no-dev

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
