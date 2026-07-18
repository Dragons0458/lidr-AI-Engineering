# Sesión 13 — grafo multiagente con gates humanos

Endpoint principal: `POST /v1/estimate/agent/graph` (bloqueante hasta gate 1).

También: `…/resume`, `…/state`, `…/stream`, `…/progress`, `…/proposal`.

## Topología

```text
START → classifier_agent ─Command→ structure_agent → human_gate_structure
  → estimate_task_hours ×N (Send) → recover_and_handover ─Command→ analysis_agent
  → human_gate_analysis ─conditional→ proposal_agent | END
```

El pipeline secuencial de 5 nodos permanece en código (`build_sequential_graph`)
solo para tests; producción usa `build_estimation_graph`.

## Reproducir

Reutilizar la transcripción compleja de la Sesión 12. El CLI hace start → resume
automático por gate (estructura aprobada + validación con propuesta):

```bash
# Stack levantado + corpus historical_task ingerido
uv run python scripts/run_agent_s13.py \
  exercises/session-12/sample_transcript_complex.txt \
  --base-url http://localhost:8000 \
  --api-key "$ESTIMATE_API_KEY" \
  --out exercises/session-13/example_graph_response.json
```

Flag `--no-proposal` omite la propuesta comercial en el gate 2.

## Evidencia esperada

| Campo | Valor |
| --- | --- |
| estimation_id / thread_id | el `estimation_id` del request |
| estado HTTP start | `200` con `state=paused`, `pending_gate.gate=structure_review` |
| estado final | `completed`, `status=validated` (con proposal si no usaste `--no-proposal`) |
| Logfire | spans `agent.graph.classifier_agent` … `proposal_agent` + gates |
| checkpoint | filas en `checkpoints` para el `thread_id` |

Artefacto de ejemplo (pre-sesión secuencial, legado):
[`example_graph_response.json`](example_graph_response.json).

Wizard Streamlit: página **Grafo Agentes** (`streamlit_ui/pages/9_Grafo_Agentes.py`).
