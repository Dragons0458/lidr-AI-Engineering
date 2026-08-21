# Session 15 decisions

Live-session register. ADRs stay the durable form; this file is the “why”
and what is still unconfirmed.

| # | Decision | Why | Still open |
| --- | --- | --- | --- |
| 1 | One Postgres with pgvector (relational + vectors + LangGraph checkpoints) | One engine to operate; matches [ADR-0002](adr/0002-single-postgres-pgvector.md) | Pool sizing (see [`scalability.md`](scalability.md)) |
| 2 | Destination is one VM with Compose + Caddy + systemd, not PaaS / Kubernetes | Course constraint + this repo already Compose-native | Redundancy, automated backups |
| 3 | Redis must be Redis Stack (RediSearch) | Semantic cache in `app/generation/cag/` needs the module; vanilla Redis is a silent miss | Production image is `redis-stack-server` (no RedisInsight `:8001`) |
| 4 | Liveness (`/health`) and readiness (`/health/ready`) are separate; neither calls the model | HEALTHCHECK must not spend tokens or wait on OpenAI | Migrating `/health/ready` off the sync engine (derived action, not S15) |
| 5 | `503` = a required dependency is missing; `500` = genuine failure; `502` = LLM upstream | WS2. Embedder / retriever / corpus index / OpenAI client absence used to return 500 | — |
| 6 | The UI→API contract is a JSON artefact checked in CI against generated OpenAPI | WS1. `scripts/check_contract.py` + `docs/contract/web-consumed-routes.json` | Keep the artefact in review when adding Streamlit pages |
| 7 | One image, two targets; secrets enter at runtime | [ADR-0003](adr/0003-multi-target-dockerfile.md). GHCR tags by SHA | `IMAGE_OWNER` / `GHCR_PAT` not set until CD is armed |
| 8 | Streamlit is the public layer and has **no login of its own** | [ADR-0001](adr/0001-streamlit-public-layer.md). Fine on localhost. On the internet, Caddy `basic_auth` is mandatory. Hash with `caddy hash-password`; never a plaintext password in the Caddyfile | A real IdP is out of scope |
| 9 | CI never calls the model; quality evaluation is Session 16 | Marker filter `not eval and not integration and not slow`. `TODO(S16)` block in `ci.yml` | Golden-set cadence |

## Revision of ADR-0001 (production)

Locally, `web` (Streamlit) is the single published port (`:8501`). In cloud
Compose (`docker-compose.prod.yml`) the public service is **Caddy** (`:80` /
`:443`) and `web` is internal (`ports: !reset []`). Streamlit remains the
application UI; it is no longer the internet-facing process. Consequence:
without `basic_auth` on Caddy, the estimator is open to the world.
