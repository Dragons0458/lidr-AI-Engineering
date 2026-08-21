# Documentation index

## By reader

### Engineering

| Doc | What it answers |
| --- | --- |
| [`architecture.md`](architecture.md) | Topology, auth layers, HTTP status contract |
| [`decisions.md`](decisions.md) | Session 15 live decisions and what is still open |
| [`adr/`](adr/) | ADRs 0001–0004 (public layer, single Postgres, multi-target image, Streamlit SQLite) |
| [`contract/web-consumed-routes.json`](contract/web-consumed-routes.json) | Routes the Streamlit UI actually calls |
| [`evaluation.md`](evaluation.md) | How quality is measured (golden set) and how to read the dashboard |

### Operations

| Doc | What it answers |
| --- | --- |
| [`deployment-local.md`](deployment-local.md) | Prod-like Compose on a laptop |
| [`ci-cd.md`](ci-cd.md) | Pipeline jobs, secrets, why CI needs Postgres |
| [`deploy-ec2.md`](deploy-ec2.md) | Cloud kit (written, **not executed**) |
| [`scalability.md`](scalability.md) | Audit of ceilings — documented, not implemented |
| [`runbooks/`](runbooks/) | `ai-service-no-responde`, `postgres-caido`, `rotar-clave-llm` |

### Product / UI

- Public UI (local): `http://localhost:8501`
- AI OpenAPI (dev override only): `http://localhost:8000/docs`

## Command cheatsheet

```bash
# Prod-like local stack (only :8501 on the host)
docker compose -f docker-compose.yml up --build -d

# Dev (bind mounts, :8000 / :5433 / :6379 published)
docker compose up --build

# Contract (no server, no LLM)
uv run python scripts/check_contract.py

# Offline test suite (same marker filter as CI)
uv run pytest -q -m "not eval and not integration and not slow"

# Session 16 golden-set harness (SPENDS TOKENS — not CI)
uv run python scripts/run_eval_s16.py --dry-run
uv run python scripts/run_eval_s16.py --arm both --label baseline-s16 --out evals/reports/

# Post-deploy smoke (spends tokens unless --skip-estimation)
docker compose -f docker-compose.yml exec web \
  python scripts/smoke_test_s15.py --base-url http://localhost:8501 --ai-url http://ai-service:8000

# Cloud compose config (must publish only 80/443)
APP_DOMAIN=example.test IMAGE_OWNER=demo \
  docker compose -f docker-compose.yml -f docker-compose.prod.yml config

# Corpus portability
./scripts/dump_corpus.sh
./scripts/restore_corpus.sh backups/corpus-YYYYMMDD.dump
```
