# Evaluation and observability

How this service is measured. Three different questions, three different tools:

| Tool | Question it answers | When | Spends tokens? |
| --- | --- | --- | --- |
| `GET /health` / `GET /health/ready` | Is the process (and Postgres/Redis) **alive**? | Always, including CI and HEALTHCHECK | No |
| [Smoke test S15](../scripts/smoke_test_s15.py) | Does the **frontier** respond with the right shape? | After deploy | Only the optional estimate probe |
| [Golden set S16](../evals/golden_set_s16.json) + [harness](../scripts/run_eval_s16.py) | Does it estimate **well** on the cases we care about? | On purpose, outside CI | **Yes** |
| [Observability dashboard](../streamlit_ui/pages/11_Observabilidad.py) | What is it doing **right now** (latency, cost, errors)? | Always-on in production | No (reads `request_metrics`) |

`/health` is cheap and dumb on purpose. Quality does not live there. A 200 on `/health` with a 0% `within_range_rate` is a system that is up and wrong.

## How to run the evaluation

**This spends real LLM tokens. Do not put it in CI.**

```bash
# 0. The corpus must be populated or every case abstains (a false safety win).
curl -s localhost:8000/health/ready
curl -s -H "X-API-Key: $ESTIMATE_API_KEY" localhost:8000/embeddings/index/stats

# 1. Validate the measuring stick without spending a token
uv run python scripts/run_eval_s16.py --dry-run

# 2. One case, RAG arm (cheap while iterating)
uv run python scripts/run_eval_s16.py \
  --arm rag --case S16-01 \
  --base-url http://localhost:8000 \
  --api-key "$ESTIMATE_API_KEY" \
  --out evals/reports/

# 3. Full lab A/B (both architectures, 7 cases). Default pacing is 7s
#    because both estimate routes are rate-limited at 10/minute.
uv run python scripts/run_eval_s16.py \
  --arm both --label baseline-s16 \
  --out evals/reports/
```

Environment: `ESTIMATE_API_KEY` or `AI_SERVICE_TOKEN` (header `X-API-Key`). Never commit a real key. The harness never sends `idempotency_key` so a second run is not a cache hit in disguise.

Flags: `--arm {rag,graph,both}`, `--case ID` (repeatable), `--dry-run`, `--pace-seconds`, `--timeout`, `--out DIR`, `--label`.

## How to read the report

Each run writes `evals/reports/eval_s16_<YYYYMMDD-HHMMSS>.{json,md}`. Two reports are comparable only when they share the same golden-set `sha256`.

| Metric | Meaning | If it drops |
| --- | --- | --- |
| `within_range_rate` | Fraction of **estimation** cases whose `engineer_days` landed in `acceptable_range` | The system got worse at the number. Diff the failing case ids. |
| `mean_absolute_error` | Mean \|predicted − expected\| in engineer-days | Estimates drifted. Check model / retrieval config in the environment block. |
| `abstention_correct` | All abstention cases actually abstained | Safety floor cracked: the system is inventing numbers with no analog. |
| `mean_latency_ms` / `p95_latency_ms` | Client-side wall time. **p95 is always printed next to `n`.** | A p95 with n=7 is not a p95. Wait for more samples before paging anyone. |
| `error_rate` | Non-2xx / total, excluding retried 429s | Transport/runtime. 503 on the graph arm is `skipped`, not failed. |
| `abstention_rate` | Fraction of cases where the system said "I don't know" | Sudden spike → data/corpus change, not a code change. |
| `source_recall` | Fraction of cases that cited at least one expected `budget_id` | Retrieval regression. Does **not** affect `passed`. |
| `total_cost_usd` | Real spend, joined via `X-Request-ID` against `/v1/observability/requests` | The number the invoice will match, not a guess from token counts. |

`passed` for an estimation case is **only** "not abstained AND inside the band". Source hits are a separate signal.

The graph arm has no `confidence=insufficient`. The harness uses a **proxy** (`grounded_task_ratio == 0` or `confidence=low` with no `has_match`) and labels it `abstention_signal: proxy`. If the graph "fails" S16-06, that is a product finding, not a harness bug.

The `ab.verdict` field is left `null` on purpose. Quality/cost trade-offs are a product decision (write them in `exercises/session-16/README.md`).

## How to read the dashboard

Open `http://localhost:8501` → **Observabilidad**. The page calls `GET /v1/observability/metrics` on `ai-service` (the Streamlit container never talks to Postgres).

| Signal | Reasonable default | If it spikes |
| --- | --- | --- |
| Mean latency | Order of seconds for a grounded estimate | Look at p95 and `n` first. |
| **p95 latency** | Always with `n`. Ignore p95 for n &lt; 20 | Take the `request_id` of a slow row and search it in Logfire. |
| Cost / request | Cents, not euros, on mini models | A silent cost climb is the LLMOps failure mode: the API stays 200. |
| Error rate | Near 0 | 429s are retried by the harness and should not dominate here. 503 = graph runtime. |
| Abstention rate | Stable vs the last eval's `abstention_rate` | Corpus empty, threshold moved, or a new kind of brief. |
| Cache hit rate | Low during an eval (the harness sends no idempotency key) | High during eval → you are not measuring the model. |

Retention: `METRICS_RETENTION_DAYS` (default 30) is documented; pruning is manual.

## How the golden set grows

Every real production miss becomes a new case (`S16-08`, …) with the same anchoring procedure: retrieve → sum analog component hours / 8 → band `[0.6×, 1.5×]`. Document the arithmetic in `exercises/session-16/evidence/01_anclaje_golden_set.md`. Do not let the LLM write its own expected numbers.

## Path mapping (exercise → this repo)

| Exercise (`ai-service/eval/…`) | Here |
| --- | --- |
| `eval/golden_set.json` | `evals/golden_set_s16.json` |
| `eval/run_eval.py` | `scripts/run_eval_s16.py` + `evals/production/` |
| `eval/reports/` | `evals/reports/` |
| `docs/evaluation.md` | this file |
| `task_description` + `project_id` | `transcript` (≥100 chars) |
| `expected_points` | `expected_engineer_days` |
| `X-Service-Token` | `X-API-Key` (`ESTIMATE_API_KEY` / `AI_SERVICE_TOKEN`) |
| `confidence < 0.3` | RAG: `confidence == "insufficient"`; graph: proxy above |

## Security

Reports, logs, Logfire spans and `request_metrics` rows contain ids, numbers and verdicts. They must never contain transcripts, prompts, generated text or API keys. Keys live in the environment (`.env`, not the repo).
