# Stress test CAG

Este reporte resume una ejecucion reducida del stress test contra Gemini 2.5 Flash. Se ejecuto 1 repeticion con escenarios `growing`, `pivot` y `contradiction`, tamanos de adjunto `0`, `20` y `100` KB, y longitudes `1`, `6` y `20` turnos. El CSV generado contiene 243 filas.

Comando base:

```bash
uv run python -m evals.stress.run \
  --scenarios growing,pivot,contradiction \
  --attachment-sizes 0,20,100 \
  --repeats 1 \
  --turn-counts 1,6,20 \
  --output evals/stress/results.csv
```

Las metricas nuevas viven en `evals/stress/metrics.py` porque dependen del shape de `turn_observed` y del snapshot debug de sesion, no del contrato historico de las evals golden.

## Resumen

| Scenario | Attachment KB | Rows | P50 latency ms | P95 latency ms | Total cost USD | Exact hit rate | Semantic hit rate | Mean recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| contradiction | 0 | 27 | 3624 | 4687 | 0.018779 | 0.000 | 0.000 | 0.189 |
| contradiction | 20 | 27 | 3939 | 4817 | 0.028358 | 0.000 | 0.000 | 0.417 |
| contradiction | 100 | 27 | 5729 | 6434 | 0.069041 | 0.000 | 0.000 | 0.281 |
| growing | 0 | 27 | 2005 | 3269 | 0.010361 | 0.000 | 0.000 | 0.135 |
| growing | 20 | 27 | 3784 | 5125 | 0.027027 | 0.000 | 0.000 | 0.340 |
| growing | 100 | 27 | 4959 | 5832 | 0.064526 | 0.000 | 0.000 | 0.111 |
| pivot | 0 | 27 | 4124 | 5196 | 0.018849 | 0.000 | 0.000 | 0.111 |
| pivot | 20 | 27 | 4618 | 7042 | 0.034181 | 0.000 | 0.000 | 0.111 |
| pivot | 100 | 27 | 5984 | 7354 | 0.073124 | 0.000 | 0.000 | 0.111 |

Totales globales: costo `0.344245 USD`, P50 latency `4204 ms`, P95 latency `6718 ms`, latencia maxima `8758 ms`, pass rate de `LatencyBudgetMetric(4000)` igual a `0.440`, recall medio `0.201`, y maximo `tokens_in` igual a `26257`.

## Curva 1: Latencia vs tokens

| Attachment KB | Avg tokens_in | Max tokens_in | P50 latency ms | P95 latency ms |
|---:|---:|---:|---:|---:|
| 0 | 3656 | 6754 | 3434 | 4867 |
| 20 | 7935 | 13354 | 3964 | 6834 |
| 100 | 21952 | 26257 | 5418 | 7208 |

## Curva 2: Coste acumulado vs turno

Coste acumulado usando las ejecuciones de 20 turnos sin adjunto para aislar el efecto de historial conversacional.

| Scenario | Turn 1 | Turn 3 | Turn 6 | Turn 10 | Turn 20 |
|---|---:|---:|---:|---:|---:|
| contradiction | 0.000337 | 0.001126 | 0.002839 | 0.005995 | 0.015566 |
| growing | 0.000243 | 0.000726 | 0.001643 | 0.003119 | 0.007352 |
| pivot | 0.000318 | 0.001174 | 0.003169 | 0.006698 | 0.015491 |

## Curva 3: Recall vs longitud del historial

| Scenario | N=1 | N=6 | N=20 |
|---|---:|---:|---:|
| contradiction | 1.000 | 0.276 | 0.266 |
| growing | 1.000 | 0.203 | 0.153 |
| pivot | 1.000 | 0.167 | 0.050 |

## Lectura

Mi CAG empieza a degradarse antes de romper de forma visible. No hubo excepciones ni fallos duros durante las 243 llamadas, pero el contrato de latencia ya queda comprometido: solo el 44.0% de las llamadas cumple `latency_ms <= 4000`. El adjunto domina esa degradacion. Al pasar de 0 KB a 100 KB, el promedio de `tokens_in` sube de 3656 a 21952 y la P50 de latencia sube de 3434 ms a 5418 ms. En otras palabras, el sistema sigue respondiendo, pero el usuario ya percibiria lentitud en los casos con adjuntos grandes.

La segunda degradacion importante es memoria. El recall medio global cae a 0.201, y en `pivot` baja de 1.000 en N=1 a 0.050 en N=20. Este es el riesgo mas peligroso para CAG: no falla con un error rojo, sino que empieza a olvidar hechos o a mezclar decisiones antiguas con decisiones nuevas. Un caso limite que justifica saltar a RAG seria una sesion de mas de 10 turnos con adjuntos cercanos a 100 KB y hechos que deben preservarse con precision contractual, porque ahi el contexto se llena con historial y texto repetido en vez de recuperar solo los fragmentos relevantes.
