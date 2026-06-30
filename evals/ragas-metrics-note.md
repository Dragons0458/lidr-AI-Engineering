# Sesión 11 — Baseline RAGAS

Ejecución: `uv run python scripts/eval_ragas_s11.py --metrics-only --cache evals/ragas_cache.json`
(juez LLM de OpenAI + `text-embedding-3-small`; respuestas/contextos cacheados del pipeline grounded).

## Tabla de métricas

## RAGAS generation evaluation (Session 11)

| Case | Category | faithfulness | answer_relevancy | context_precision | context_recall |
| --- | --- | --- | --- | --- | --- |
| G1 | clear | 0.583 | 0.355 | 0.917 | 0.833 |
| G2 | ambiguous | 0.973 | 0.123 | 0.897 | 0.600 |
| G3 | conflicting_sources | 0.862 | 0.358 | 0.757 | 0.000 |
| G4 | abstention | 0.143 | 0.000 | n/a | n/a |
| G5 | baseline | 0.800 | 0.474 | 0.750 | 0.222 |
| **AVG** | — | 0.672 | 0.262 | 0.830 | 0.414 |

> `n/a` en G4: para una abstención correcta no hay afirmaciones de referencia que recuperar,
> así que context_precision/recall no aplican y se excluyen del promedio en lugar de puntuar 0.

## La nota: lo que más chirría

- **G3 (conflicting_sources) tiene context_recall = 0.000** pese a un context_precision sano (0.757):
  la recuperación trae chunks relevantes pero **no todos** los que la referencia necesita (las dos
  fuentes de telemedicina que se contradicen). Es el patrón "precision alta + recall bajo" = **hueco de
  recuperación**, no de generación.
- **answer_relevancy es flojo de media (0.262)** y baja sobre todo en G2 (ambiguo, 0.123): cuando el
  brief está poco especificado, la respuesta se dispersa y las preguntas que el juez reconstruye divergen
  de la original.
- **faithfulness es la métrica más alta (AVG 0.672)** pero **ruidosa**: el juez es un LLM, así que estos
  números valen como **comparables entre versiones**, no como notas absolutas.
