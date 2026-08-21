# Session 15 — failure modes (Block 1)

Five fixtures with the defect **inside the file** and the why in the header.
None of them is meant to be run except #1 (to watch the build fail).

Use them live: show the symptom → open the fixture → fix it → compare with
the real file (`Dockerfile`, `docker-compose.yml`, `.env.example`).

| # | File | Symptom | Cause (this repo) |
| --- | --- | --- | --- |
| 1 | `01-image-does-not-build.Dockerfile` | `"/app/uv.lock": not found` or a glacial rebuild | `COPY . .` **before** `uv sync`, single stage, no `target:` — every code edit reinstalls torch |
| 2 | `02-wrong-boot-order.yml` | `ai-service` dies on the first `up`, lives on the second | `depends_on` as a list waits for **creation**, not **readiness**; `migrate` is not `service_completed_successfully` |
| 3 | `03-localhost-vs-service-name.yml` | UI loads, every call is `Connection refused localhost:8000` | Inside `web`, `localhost` is that container. `ESTIMATION_API_BASE_URL` must be `http://ai-service:8000/api/v1` |
| 4 | `04-ports-leak.yml` | Everything works — that is the bug | `ports: ["8000:8000"]` on `ai-service` breaks the frontier. The token does **not** replace a closed port |
| 5 | `05-token-mismatch.env` | Stack `healthy`, 401 everywhere | `AI_SERVICE_TOKEN` ≠ `ESTIMATE_API_KEY`. Silent variant: `require_service_token` **fails closed** if no token is set |

## Order

1. **While building** — fixture 1.
2. **While starting** — fixtures 2 and 3.
3. **Once it "works"** — fixtures 4 and 5.

```bash
docker build -f exercises/session-15/failure_modes/01-image-does-not-build.Dockerfile .
uv run pytest -q tests/test_failure_modes_s15.py -v
```

The tests pin **both halves**: the defect is still in the fixture, and the
real file does not have it.
