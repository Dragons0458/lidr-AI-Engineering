# Session 16 — quality and observability

Production measuring stick (golden set + harness) and always-on vital signs
(instrumentation + dashboard). **This spends tokens** when you run the harness
for real; `--dry-run` does not.

## Reproduce

```bash
uv run python scripts/run_eval_s16.py --dry-run

uv run python scripts/run_eval_s16.py \
  --arm both --label baseline-s16 \
  --base-url http://localhost:8000 \
  --api-key "$ESTIMATE_API_KEY" \
  --out evals/reports/
```

Then open `http://localhost:8501` → Observabilidad.

Full operator guide: [`docs/evaluation.md`](../../docs/evaluation.md).
Anchoring audit trail: [`evidence/01_anclaje_golden_set.md`](evidence/01_anclaje_golden_set.md).

## Hallazgos para el directo

Filled after `--arm both --label baseline-s16` (run_id `14071272`,
2026-08-21, golden sha256 `35d3c3c503bc…`). Environment: generation
`gemini/gemini-2.5-flash`, graph/recovery `gpt-4o-mini` (this OpenAI org cannot
call `gpt-5` / `gpt-5-mini`). Corpus was populated (`total_chunks=1596`).

- Cases the **RAG** arm fails (id + why): **all seven**. Estimation cases
  S16-01..05 overshoot the band by 5–20× (e.g. S16-01 352 vs 61 [37, 92];
  S16-03 710 vs 67). S16-06/07 did not abstain (`confidence` was not
  `insufficient`) and invented 264 / 494 days. `within_range_rate=0%`,
  `abstention_correct=False`, MAE 440 engineer-days. Source recall 40%
  (hits on S16-03 and S16-04 only) — the number is wrong even when a
  expected `budget_id` is cited.
- Cases the **graph** arm fails (id + why): S16-01 (proxy abstention — no
  hours, so it refused a case that *should* estimate). S16-02..05
  underestimate (12 / 16 / 16 / 3 vs bands around 29–68). S16-06 invented
  4 days instead of abstaining. Only S16-07 passed (proxy abstention on
  the empty brief). `within_range_rate=0%`, MAE 38 days (an order of
  magnitude closer than RAG, still outside every estimation band).
- Abstention gap (S16-06 / S16-07): harness `abstention_gap=['S16-01','S16-07']`.
  RAG never abstains. Graph proxy fired on S16-01 (false safety: a
  well-anchored portal) and S16-07 (correct), and **did not** fire on
  S16-06 (quantum-metrology no-precedent, predicted=4). Confirming the
  article-03 finding: the graph has no `confidence=insufficient`, and the
  proxy is not a substitute.
- Dashboard signal that stands out (p95, cost/request, abstention rate):
  Observabilidad after the run (n=48 in the 24h window): mean 9506 ms,
  **p95 27812 ms**, error rate 8.3% (early graph 502s on gpt-5-mini),
  abstention 12.5%, cache hit 0% (harness sends no idempotency key).
  Cost/request on the dashboard ($0.0004) is lower than the eval join
  (~$0.0014 RAG / $0.0020 graph) because GET `/state` rows dilute the mean.
  p95 with n=48 is starting to be readable; n=7 in the report is not.
- Quality vs cost (which arm do we ship, and what do we give up): **neither
  as-is**. RAG is cheap and fast (~13 s, $0.010 the set) but systematically
  inflates and never refuses. Graph is ~3× slower (~38 s) and ~1.4× costlier
  for the set, closer on MAE, but still 0% in-range and a broken safety
  floor. Ship a calibrated hours consensus (graph) only after the graph
  learns to abstain; do not ship RAG `from-transcript` numbers until the
  generation model stops treating engineer-days as story-point-scale
  fiction. `ab.verdict` stays unset on purpose.

## Evidence

| File | What it proves |
| --- | --- |
| `evidence/00_preflight.txt` | Corpus + ready + models. Without this, later numbers are not defensible. |
| `evidence/01_anclaje_golden_set.md` | Why the expected days are those days. |
| `evidence/02_primer_caso.txt` | First real `--case S16-01` (tokens). |
| `evidence/03_abstencion_dos_brazos.txt` | `--arm both --case S16-06`. |
| `evidence/04_informe_completo.md` | Copy of a full `eval_s16_*.md`. |
| `evidence/05_dashboard.png` | Explicit deliverable of the exercise. Capture after a full run. |
| `evidence/06_logfire_trace.txt` | Optional; skipped here (`LOGFIRE_TOKEN` unset). |
