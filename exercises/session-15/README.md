# Session 15 — Production dockerisation evidence

Manifest for the Session 15 pre-work: one public surface (`web`), private
`ai-service` / `postgres` / `redis`, secrets at runtime, named volumes, and
ordered healthchecks — plus **check 4**: a real end-to-end estimate through the
frontier.

## Topology

See [`docs/architecture.md`](../../docs/architecture.md) and ADRs under
[`docs/adr/`](../../docs/adr/).

## How to reproduce

```bash
cp .env.example .env   # real OPENAI_API_KEY
docker compose -f docker-compose.yml up --build -d
```

Dev (ports + reload): `docker compose up --build`.

## Evidence files

| File | Check |
| --- | --- |
| `evidence/00_before_ports.txt` | Host could reach `:8000` before the frontier |
| `evidence/00_before_image.txt` | Image sizes before multi-target split |
| `evidence/01_compose_ps.txt` | Four services healthy |
| `evidence/02_web_host.txt` | Streamlit healthy from the host |
| `evidence/02_web_host.png` | UI home from the browser |
| `evidence/03_frontier.txt` | Host `:8000` refused; in-network OK |
| `evidence/04a_corpus_count.txt` | Chunk counts before spending tokens |
| `evidence/04_e2e_run.txt` | End-to-end estimate through `web → ai-service → pgvector` (shape asserts) |
| `evidence/04_e2e_ui.png` | Same path from the Streamlit UI (confidence + sources) |
| `evidence/05_restart.txt` | Chunk + estimation row counts equal before/after `down`/`up` |
| `evidence/05_auth_matrix.txt` | Routes without auth vs allowlist |

## Docs

- [`docs/deployment-local.md`](../../docs/deployment-local.md) — five checks
- Runbooks under [`docs/runbooks/`](../../docs/runbooks/)
- ADRs under [`docs/adr/`](../../docs/adr/)
- [`docs/architecture.md`](../../docs/architecture.md) — topology
