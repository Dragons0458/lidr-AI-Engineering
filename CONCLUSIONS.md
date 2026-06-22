# Sesión 10 — Conclusiones de la evaluación de recuperación

## Cómo leer la tabla: qué es `precision@5`

`precision@5` mide la calidad de los **5 chunks que el retriever pone arriba** — justo los
que el pipeline entrega al LLM (`RERANK_TOP_N = 5`). De esos 5, ¿cuántos son **relevantes**?

```
hits = nº de chunks del top-5 cuyo budget_id ∈ relevant_budget_ids
precision@5 = hits / 5
```

Un chunk cuenta como acierto **sii** su `budget_id` está en la lista anotada de esa consulta
en el golden set. Así, `0.80` de media = en promedio **4 de cada 5** resultados del top-5
son presupuestos que el golden set considera relevantes. Mide la **calidad de lo que llega
al generador**, no la calidad de la estimación final (ver «Lo que esta medición no da»).

## Tabla comparativa (configs A–D)

Medida con `scripts/eval_retrieval_s10.py` sobre `evals/golden_retrieval.json`
(5 consultas generadas y anotadas con asistencia de IA sobre el corpus
`data/budgets_sample.json`, 30 chunks ingeridos).
La latencia de embedding queda excluida; la latencia de recuperación es la **mediana**
de 3 ejecuciones medidas por (consulta, config) tras descartar una de calentamiento.

| Config | Búsqueda | Reranking | precision@5 | Latencia mediana (ms) |
| --- | --- | --- | --- | --- |
| **A** | Vectorial | No | 0.48 | 4.0 |
| **B** | Híbrida | No | **0.52** | 5.4 |
| **C** | Vectorial | Sí | 0.40 | 186.5 |
| **D** | Híbrida | Sí | 0.40 | 186.8 |

### precision@5 por consulta

| Consulta | A | B | C | D |
| --- | --- | --- | --- | --- |
| Q1 (`Stripe`, keyword) | 0.20 | **0.40** | 0.20 | 0.20 |
| Q2 (`SCADA`, keyword) | 0.20 | 0.20 | **0.40** | **0.40** |
| Q3 (`telemedicine`, keyword) | **0.80** | **0.80** | 0.40 | 0.40 |
| Q4 (`Stripe marketplace`) | **0.40** | **0.40** | 0.20 | 0.20 |
| Q5 (telemedicina NL larga) | 0.80 | 0.80 | 0.80 | 0.80 |

El golden set mezcla **consultas keyword** (Q1–Q4) y una **consulta NL larga** (Q5).
Las keyword exponen tokens exactos del corpus (`Stripe`, `SCADA`) donde FTS y reranking
pueden mover el ranking; Q5 sirve de sanity check. Con solo 30 chunks (2 por presupuesto)
es un laboratorio de mecanismos, no una validación estadística de producción.

## Lectura por consulta

- **Q1 — la híbrida gana.** `Stripe` matchea léxicamente dos presupuestos e-commerce; la
  config A con recall estrecho (`top_k=5`) suele traer solo un chunk de Stripe, mientras B
  fusiona FTS + vector con recall amplio y sube de 0.20 a 0.40.
- **Q2 — el reranker gana.** `SCADA` confunde al bi-encoder con telemetría industrial
  genérica (IND-001/IND-003); el cross-encoder promueve IND-004 (parque eólico).
- **Q3 y Q4 — el reranker empeora.** El bi-encoder ya rankeaba bien; el cross-encoder
  (`mmarco-mMiniLMv2`) reordena a presupuestos adyacentes pero incorrectos (farmacia HLT-004,
  otros e-commerce). **Más reranking no implica más precisión** en corpus pequeños.
- **Q5 — baseline.** Consulta NL larga donde las cuatro configs empatan — sanity check, no
  diferenciador.

## Decisión

Para este estimador, mantendría por defecto **B (híbrida, sin rerank)**: es la config con
mejor precision@5 agregada (**0.52**) con latencia casi idéntica a A (~5 ms vs ~4 ms). El
reranker añade **~180 ms** por recuperación y **baja** la media a 0.40 en este golden set
(retrocesos en Q3/Q4 compensan la mejora en Q2).

La híbrida sin rerank es el punto dulce aquí: FTS con `websearch_to_tsquery('english', …)`
rescata presupuestos por token exacto sin pagar el coste del cross-encoder. Activaría **C o D**
solo si en logs de producción aparecen casos tipo Q2 (keyword ambiguo donde el reranker sí
ayuda) y no casos tipo Q3/Q4.

Los flags ya van por entorno (`RETRIEVAL_SEARCH_MODE`, `RERANKER_ENABLED`), así que encenderlos
es un experimento, no un refactor.

## Lo que esta medición no da

- **5 consultas no tienen potencia estadística** — en tráfico real haría falta un set mayor
  y anotación manual.
- **Sesgo de la anotación** — consultas y etiquetas producidas con asistencia de IA.
- **Solo recuperación** — precision@5 dice qué llega al LLM, no si la estimación es buena.

## Notas técnicas (D2)

La búsqueda full-text usa la configuración de PostgreSQL **`english`**, no `spanish`,
porque el texto de los chunks en `data/budgets_sample.json` se renderiza en inglés
técnico (`render_component_text`). Con `spanish` se aplicarían stemmer y stopwords
equivocados. El `regconfig` coincide en la migración `0003_session10_fts` y en
`ChunkStore.search_lexical` (`websearch_to_tsquery('english', …)` + `ts_rank`).
