# Session 14 — Supervisor multi-agent evidence manifest

This kit proves the Session 14 pre-work: a hand-built supervisor (`StateGraph` +
`Command`), minimum tool privilege with an enforceable guard, and a signal-driven
human review gate.

## What to demonstrate

1. **Routing** — the model chooses the next agent; each decision lands in
   `routing_history` with `reason` and `source` (`llm` | `fallback` | `limit`).
2. **Three deterministic brakes** — step budget, legality guard, dependency-ladder
   fallback. Unit tests run offline because the fallback path needs no network.
3. **Privilege** — each specialist only reaches its allowlisted tools via
   `guarded_dispatch`. Denials are audited (`outcome="denied"`).
4. **HITL** — `requires_human_review` is a pure signal; `human_review_gate` calls
   `interrupt()` before any state write. Resume accepts `approve` | `adjust` |
   `reject`.

## Transcripts

| File | Purpose |
| --- | --- |
| `sample_transcript_happy_path.txt` | Well-grounded supplier portal — completes without pause |
| `sample_transcript_edge_case.txt` | Exotic QKD / COBOL / iris stack — should trip the human gate |

## How to run

```bash
# Regenerate the three deterministic artifacts (no network, DB, or API key)
uv run python scripts/run_agent_s14.py --generate-evidence

# Offline happy path (MemorySaver + deterministic collaborators)
uv run python scripts/run_agent_s14.py \
  exercises/session-14/sample_transcript_happy_path.txt \
  --memory --stub \
  --out exercises/session-14/example_run_happy.txt

# Edge-case with auto-approve on the gate
uv run python scripts/run_agent_s14.py \
  exercises/session-14/sample_transcript_edge_case.txt \
  --memory --stub --decision approve \
  --out exercises/session-14/example_run_edge_case.txt

# Level-3 denial demo (budget_searcher reaches for validate_estimate once)
uv run python scripts/run_agent_s14.py \
  exercises/session-14/sample_transcript_happy_path.txt \
  --memory --stub --violate \
  --out exercises/session-14/example_run_violate.txt

# Live HTTP: start -> optional review resume -> final checkpoint
uv run python scripts/run_agent_s14.py \
  exercises/session-14/sample_transcript_edge_case.txt \
  --base-url http://localhost:8000 \
  --api-key "$ESTIMATE_API_KEY" \
  --decision approve
```

`--transcript PATH` remains available as a backwards-compatible alternative to
the positional transcript. Offline mode deliberately forces the production
fallback ladder so the evidence is reproducible and requires no model call.

## Generated evidence

| Artifact | Assertion |
| --- | --- |
| `example_run_happy.txt` | All agents route in order, grounding is sufficient, no review pause |
| `example_run_edge_case.txt` | No precedent + low confidence + high-risk scope trigger review and expose risk flags |
| `example_run_violate.txt` | A forbidden `budget_searcher -> validate_estimate` attempt is denied and audited |
| `example_run_happy_http.txt` | Live API/model run `s14-live-happy-20260720-2217`: LLM routing, confidence 0.80, no pause |
| `example_run_edge_case_http.txt` | Live API/model run `s14-live-edge-20260720-2218`: risk-driven pause, approve, completed |

The first three artifacts prove the deterministic control plane. The HTTP
artifacts prove the configured model, API auth, start/resume/state contract,
and Postgres checkpointing. Both live checkpoints were read as
`completed/validated` after restarting the API process. Logfire was configured
with `send_enabled=false` during this acceptance, so there is intentionally no
external trace URL; the local routing and audit trails remain in the artifacts.

HTTP surface (auth `ESTIMATE_API_KEY`):

- `POST /v1/estimate/agent/supervisor`
- `POST /v1/estimate/agent/supervisor/{id}/resume`
- `GET  /v1/estimate/agent/supervisor/{id}/state`

`thread_id` is namespaced as `s14:{estimation_id}` so it never collides with the
Session 13 graph on the shared checkpointer.

## Tool privilege table

| Agent | Tools |
| --- | --- |
| `supervisor` | (none — routes only) |
| `requirements_extractor` | (none — model only) |
| `budget_searcher` | `search_budgets` |
| `estimate_generator` | `calculate_estimate` |
| `coherence_validator` | `validate_estimate` |

Unlike the course reference, this repo keeps the real Session 12
`calculate_estimate` tool (median + 15% contingency). There is no
`derive_task_hours` alias.
