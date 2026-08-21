# CI/CD

The pipeline lives in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
Each job answers one question.

| Job | Question |
| --- | --- |
| `changes` | Did this push touch the AI service, the UI, or the deploy kit? |
| `test` | Does the offline suite still pass, formatted and linted? |
| `contract` | Do the routes Streamlit calls still exist in generated OpenAPI? |
| `build` | Do both Dockerfile targets (`ai-service`, `web`) still pack? |
| `deploy` | Can we roll this SHA onto the VM? **Written, switched off** (`vars.CD_ENABLED`). |

```mermaid
flowchart LR
  changes --> test --> contract --> build --> deploy
```

## CI never calls the model

`uv run pytest -q -m "not eval and not integration and not slow"` is the gate.
Golden-set / RAGAS jobs are commented as `TODO(S16)`.

Doubles actually used in this repo:

| Area | Double |
| --- | --- |
| Redis / CAG | `fakeredis` (`tests/conftest.py`) |
| RAG retrieval / stages | monkeypatch of `app.generation.rag.*` and router `get_embedder` |
| Session 12 agent loop | scripted fake `AsyncOpenAI` in `tests/unit/generation/agentic/` |
| Graph / supervisor HTTP | fake runtime injected on `app.state` |

Reproduce the CI environment on a laptop (empty env, do **not** inherit `.env`):

```bash
env -i PATH="$PATH" HOME="$HOME" \
  OPENAI_API_KEY=sk-test-not-a-real-key APP_ENV=development \
  ESTIMATE_API_KEY=ci-estimate-key AI_SERVICE_TOKEN=ci-estimate-key \
  RETRIEVAL_API_KEY=ci-retrieval-key \
  DATABASE_URL=postgresql+psycopg://estimator:estimator@localhost:5433/estimator \
  uv run pytest -q -m "not eval and not integration and not slow"
```

`tests/conftest.py` fills gaps with `setdefault` **before** importing Settings.
`APP_ENV=test` is invalid (`development|staging|production`) and would crash
on import. The deepeval pytest plugin calls `load_dotenv()`; the hermetic
fixture keeps `APP_ENV` legal so a leaked `.env` cannot take the suite down.

## Why CI starts Postgres

Several modules enter the FastAPI `lifespan`, which opens the LangGraph
checkpointer pool (`app/foundation/persistence/langgraph.py`). Without a
reachable Postgres every `TestClient` blocks on the pool timeout — tests
still pass, they just take minutes. The `test` job therefore runs
`pgvector/pgvector:pg16` on `localhost:5433`.

## Secret chain

| Secret | Where it lives | Travels in CI? |
| --- | --- | --- |
| `OPENAI_API_KEY` | `/opt/estimador-cag/.env` (`chmod 600`) | **No** |
| `AI_SERVICE_TOKEN` / `ESTIMATE_API_KEY` | same `.env` | **No** (CI uses placeholders) |
| `RETRIEVAL_API_KEY` | same `.env` | **No** |
| `POSTGRES_PASSWORD` | same `.env` | **No** |
| `GHCR_PAT` | GitHub Actions secret | Yes, only to `docker/login-action` |
| `EC2_HOST` / `EC2_USER` / `EC2_SSH_KEY` | GitHub Actions secrets | Yes, only the `deploy` job |
| `CD_ENABLED` / `APP_DOMAIN` / `GHCR_OWNER` | GitHub Actions **variables** | Yes |

Images are tagged with `${{ github.sha }}` **and** `latest`. Rollback is
`IMAGE_TAG=<previous-sha>` plus `compose pull && up`, not a rebuild.

## Turning CD on

The `deploy` job is already written. To arm it:

1. Provision the VM (see [`deploy-ec2.md`](deploy-ec2.md)) — **not done**.
2. Set GitHub Environment `production` secrets `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`.
3. Set repository variable `CD_ENABLED=true` and `APP_DOMAIN`.
4. Put application secrets on the host by `scp`, never in the pipeline.

Until those exist, `deploy` is skipped on every push.
