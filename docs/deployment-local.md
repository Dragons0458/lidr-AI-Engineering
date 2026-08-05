# Local deployment (Session 15)

## Prerequisites

- Docker Desktop / Engine with Compose v2
- `.env` from `.env.example` with a real `OPENAI_API_KEY` (or other provider)
- `ESTIMATE_API_KEY` / `AI_SERVICE_TOKEN` and `RETRIEVAL_API_KEY` set (demo values
  in `.env.example` are fine for local)
- Seeded vector corpus (`budget_chunks` count > 0) for check 4 — see Seed below

## Commands

**Prod-like (evidence / frontier demo)** — base file only, no override:

```bash
cp .env.example .env   # once; put a real LLM key
docker compose -f docker-compose.yml up --build -d
```

**Dev** — base + override (bind mounts, `--reload`, published `8000`/`5433`):

```bash
docker compose up --build
```

Stop without wiping volumes:

```bash
docker compose -f docker-compose.yml down
```

## Five checks (Paso 7)

### 1. Four services healthy

```bash
docker compose -f docker-compose.yml ps --format "table {{.Service}}\t{{.Status}}"
```

Expect `redis`, `postgres`, `ai-service`, `web` as `Up … (healthy)`.
`migrate` should be `Exited (0)`.

### 2. Frontend from the host

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://localhost:8501/_stcore/health
```

Expect `200`. Open `http://localhost:8501` in a browser
(evidence: `exercises/session-15/evidence/02_web_host.png`).

### 3. AI service not on the host

```bash
curl -sS --max-time 3 http://localhost:8000/health; echo " exit=$?"
```

Expect connection refused (`exit=7`). From inside the network:

```bash
docker compose -f docker-compose.yml exec web \
  python -c "import httpx; print(httpx.get('http://ai-service:8000/health').json())"
```

Expect `{'status': 'ok', ...}`.

> **Note on `/docs` and `/openapi.json`:** they stay on the service-token
> allowlist so OpenAPI works in **dev** (override publishes `:8000`). In
> prod-like they are unreachable from the host — not a hole, the port is closed.

### 4. End-to-end estimation path

**Primary (what the exercise asks):** from the Streamlit UI at
`http://localhost:8501`, open **RAG Estimación** and run an estimate with a
real transcript (e.g. `exercises/session-14/sample_transcript_happy_path.txt`).
Path: `web → ai-service (X-API-Key) → postgres/pgvector → response` with
confidence and sources visible
(evidence: `exercises/session-15/evidence/04_e2e_ui.png`).

**Automated smoke (same path, spends LLM tokens — post-deploy only, never CI):**

```bash
# Requires corpus (check 4a). Idempotent key avoids a second bill on retry.
docker compose -f docker-compose.yml exec web python scripts/smoke_test_s15.py
```

Must print `confidence` ∈ {high,medium,low}, `sources=N` with N>0, and
`days=…`. If you see `insufficient`, seed the corpus first — that is not a
frontier failure.

### 5. Restart keeps data

Count **rows**, not just volumes:

```bash
docker compose -f docker-compose.yml exec postgres \
  psql -U estimator -d estimator -c "select count(*) from budget_chunks;"
docker compose -f docker-compose.yml exec web \
  python -c "import sqlite3; print(sqlite3.connect('/app/streamlit_ui/data/frontend.db').execute('select count(*) from estimations').fetchone())"

docker compose -f docker-compose.yml down          # no -v
docker compose -f docker-compose.yml up -d

# Repeat the two counts — they must match.
```

Contrast: `down -v` deletes volumes on purpose.

## Seed / corpus

If RAG search is empty after a fresh volume, re-ingest from the AI container
(with scripts present in the `ai-service` image):

```bash
docker compose -f docker-compose.yml exec postgres \
  psql -U estimator -d estimator -c \
  "select 'budget_chunks' t, count(*) from budget_chunks;"

docker compose -f docker-compose.yml exec ai-service \
  python scripts/build_task_corpus.py --ingest
```

See also Session 8–10 ingest notes in the root `README.md`.
