# Estimador CAG

FastAPI + Streamlit application that generates software project effort estimations from meeting summaries using LLMs through LiteLLM.

## Quick Start (Docker Compose)

1. Create env file:

```bash
cp .env.example .env
```

2. Edit `.env` with a real API key (OpenAI, Anthropic, or Google).

   - For Docker Compose, `streamlit` reads `ESTIMATION_API_BASE_URL` from environment.
   - A `.streamlit/secrets.toml` file is optional and not required.

3. Build and run everything:

```bash
docker compose up --build
```

4. Open URLs:

- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Streamlit UI: `http://localhost:8501`

> Note: health is exposed at `/health` (root), not at `/api/v1/health`.

5. Optional quick check:

```bash
curl http://localhost:8000/health
```

Stop services:

```bash
docker compose down
```

## Features

- REST API for project estimation generation.
- Optional streaming estimation endpoint.
- Prompt versioning (`v1`, `v2`) with Jinja templates.
- Structured request/response schemas with Pydantic validation.
- Cost and token usage reporting based on model pricing rules.
- Streamlit UI for interactive testing and demo usage.
- Prompt rendering tests with `pytest`.

## Tech Stack

- Python `3.11+`
- FastAPI
- LiteLLM
- Jinja2
- Streamlit
- Structlog
- Pytest
- UV (project lockfile included as `uv.lock`)

## Project Structure

```text
app/
  main.py                        # FastAPI app entrypoint
  config.py                      # Environment-based settings
  routers/estimations.py         # API endpoints
  services/estimation_service.py # LLM call + streaming call
  formatters/llm_formatters.py   # Maps LLM output to API response
  schemas/estimation.py          # Request/response and enums
  prompts/
    loader.py                    # Jinja rendering for prompt versions
    estimation/
      v1/
      v2/
streamlit_app.py                 # Streamlit frontend
tests/prompts/test_estimation_v1.py
```

## Requirements

- Python `>=3.11`
- One provider API key based on selected `LLM_PROVIDER`:
  - OpenAI -> `OPENAI_API_KEY`
  - Anthropic -> `ANTHROPIC_API_KEY`
  - Google -> `GOOGLE_API_KEY`

## Installation

Using `uv` (recommended):

```bash
uv sync
```

If you need development dependencies:

```bash
uv sync --group dev
```

## Environment Variables

Create a `.env` file in the project root:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here
APP_ENV=development
LOG_LEVEL=DEBUG
```

For other providers:

- Set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=...`
- Set `LLM_PROVIDER=google` and `GOOGLE_API_KEY=...`

## Run the API

```bash
uv run uvicorn app.main:app --reload
```

Default local endpoints:

- `GET /health`
- `POST /api/v1/estimate`
- `POST /api/v1/estimate/stream`

## Run the Streamlit App

```bash
uv run streamlit run streamlit_app.py
```

By default, the Streamlit app targets `http://localhost:8000/api/v1`.

You can override this via Streamlit secret key:

- `ESTIMATION_API_BASE_URL`

## API Request Model

`POST /api/v1/estimate?prompt_version=v1|v2`

Request body shape:

```json
{
  "transcript": "Meeting summary text...",
  "project_type": "web_saas",
  "detail_level": "medium",
  "output_format": "line_items",
  "reference_projects": [
    {
      "name": "Billing MVP",
      "summary": "Project focused on subscriptions and invoicing.",
      "estimated_hours": 280,
      "team": "2 backend, 1 frontend",
      "outcome": "Released in 8 weeks"
    }
  ]
}
```

Enums:

- `project_type`: `mobile_app`, `web_saas`, `internal_tool`, `data_pipeline`
- `detail_level`: `summary`, `medium`, `detailed`
- `output_format`: `phases_table`, `line_items`, `narrative`

## Response Model

```json
{
  "estimation": "...",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "timestamp": "2026-05-13T23:00:00.000000",
  "usage": {
    "tokens_used": 1234,
    "cost_estimate": 0.0009
  },
  "prompt_version": "v1"
}
```

## Prompt Versions

- `v1`: classic estimation instructions with concise planning outputs.
- `v2`: risk-aware planning style, includes buffer-hour guidance and stronger risk/dependency emphasis.

Prompt templates are located at:

- `app/prompts/estimation/v1/`
- `app/prompts/estimation/v2/`

Rendering is handled by `app/prompts/loader.py`.

## Testing

Run tests:

```bash
uv run pytest
```

Current tests focus on prompt rendering behavior and version-specific instructions.
