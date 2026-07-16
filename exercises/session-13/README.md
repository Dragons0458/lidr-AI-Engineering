# Sesión 13 — Estimación secuencial con LangGraph (pre-sesión)

Endpoint entregable: `POST /v1/estimate/agent/graph`

## Topología

```text
START
  → extract_requirements
  → classify_components
  → search_budgets          # secuencial (un componente cada vez)
  → generate_estimate
  → validate_and_consolidate
  → END
```

## Reproducir

Reutilizar la transcripción compleja de la Sesión 12:

```bash
# Stack levantado + corpus historical_task ingerido
uv run python scripts/run_agent_s13.py \
  exercises/session-12/sample_transcript_complex.txt \
  --base-url http://localhost:8000 \
  --api-key "$ESTIMATE_API_KEY" \
  --out exercises/session-13/example_graph_response.json
```

## Manifiesto de evidencia

| Campo | Valor |
| --- | --- |
| estimation_id / thread_id | `s13-acceptance-final` |
| comando | `uv run python scripts/run_agent_s13.py exercises/session-12/sample_transcript_complex.txt --estimation-id s13-acceptance-final --out exercises/session-13/example_graph_response.json` |
| estado HTTP | `200` |
| status del grafo | `needs_review` |
| número de componentes | `16` (1 con presupuesto, 15 supuestos sin referencias) |
| total_hours | `51.2` |
| Logfire | solo spans locales (`LOGFIRE_TOKEN` vacío → `send_to_logfire=if-token-present`); cinco spans `agent.graph.*` observados en los logs de la API; sin URL remota |
| checkpoint presente | sí — 7 filas en `checkpoints` para `thread_id=s13-acceptance-final` |

Artefacto de respuesta: [`example_graph_response.json`](example_graph_response.json).

En esta corrida se observó el waterfall serial de `search_budgets` (un span de
componente tras otro) en el log de la API.
