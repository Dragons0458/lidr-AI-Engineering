# Session 11 — RAGAS generation baseline

Reference metrics for ``scripts/eval_generation_s11.py --gate``. Numbers taken from
``eval_ragas_s11.py`` output documented in ``evals/ragas-metrics-note.md``.

## Averages (gate thresholds)

| Metric | Baseline | Tolerance |
| --- | --- | --- |
| faithfulness | 0.672 | 0.08 |
| answer_relevancy | 0.262 | 0.08 |
| context_precision | 0.830 | 0.08 |
| context_recall | 0.414 | 0.08 |

G4 (abstention) is excluded from precision/recall averages.

## Per-case snapshot

| Case | faithfulness | answer_relevancy | context_precision | context_recall |
| --- | --- | --- | --- | --- |
| G1 clear | 0.583 | 0.355 | 0.917 | 0.833 |
| G2 ambiguous | 0.973 | 0.123 | 0.897 | 0.600 |
| G3 conflicting | 0.862 | 0.358 | 0.757 | 0.000 |
| G5 baseline | 0.800 | 0.474 | 0.750 | 0.222 |

Machine-readable: ``evals/ragas_baseline_s11.json``.
