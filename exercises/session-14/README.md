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

### Edge-case fixtures (one HITL signal each)

| File | Signal under test |
| --- | --- |
| `edge_cases/low_confidence.txt` | Vague / open scope → `low_confidence` |
| `edge_cases/no_precedent.txt` | No historical analogs → `no_precedent` |
| `edge_cases/out_of_historical_range.txt` | “Typical” modules at extreme scale → `out_of_range` |

Pinned by `tests/unit/generation/agentic/graph/test_supervisor_hitl_edge_cases.py`
(offline fakes force the signal deterministically).

### Failure-mode demos

| Module | Symptom | Fix |
| --- | --- | --- |
| `failure_modes/routing_no_converge.py` | Router ping-pong until step budget | `guard=True` restores `_already_ran` |
| `failure_modes/state_clobber.py` | Parallel writes clash on a plain list | `Annotated[..., operator.add]` |
| `failure_modes/interrupt_no_resume.py` | Resume on wrong `thread_id` | `s14:{estimation_id}` on start and resume |

See `failure_modes/README.md`. Pinned by `test_supervisor_failure_modes.py`.

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

# Level 2 LIVE — competition pattern (conservative vs aggressive + synthesizer range)
uv run python scripts/run_agent_s14.py \
  exercises/session-14/sample_transcript_happy_path.txt \
  --memory --stub --compete \
  --out exercises/session-14/example_run_competition.txt

# Level 3 LIVE — sandboxed persistence (irreversible save_estimate after human approve)
uv run python scripts/run_agent_s14.py \
  exercises/session-14/sample_transcript_happy_path.txt \
  --memory --stub --persist --decision approve \
  --out exercises/session-14/example_run_persistence.txt

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
| `example_run_competition.txt` | Conservative/aggressive proposals diverge; synthesizer range; gate trips on `high_divergence` |
| `example_run_persistence.txt` | `DEFER` before approve, then `persistence = persisted` after the gate |
| `example_run_happy_http.txt` | Live API/model run `s14-live-happy-20260720-2217`: LLM routing, confidence 0.80, no pause |
| `example_run_edge_case_http.txt` | Live API/model run `s14-live-edge-20260720-2218`: risk-driven pause, approve, completed |

The first five artifacts prove the deterministic control plane (including LIVE
competition and persistence). The HTTP artifacts prove the configured model, API
auth, start/resume/state contract, and Postgres checkpointing. Both live
checkpoints were read as `completed/validated` after restarting the API process.
Logfire was configured with `send_enabled=false` during this acceptance, so there
is intentionally no external trace URL; the local routing and audit trails remain
in the artifacts.

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
