# Sesión 11 — Informe de verificación de citaciones (ejemplo)

**Pipeline:** `estimate_from_transcript` (ruta grounded, Sesión 9/11)
**Petición:** brief de e-commerce con checkout de Stripe (alineado con el caso golden G1)
**Fecha:** 2026-06-28

## Contexto recuperado

5 chunks de presupuesto retenidos tras el presupuesto de tokens (presupuestos principales: `S07-ECO-001`, `S07-ECO-002`). Cada bloque `<source>` incluye `id` y `document_id` para la atribución a nivel de línea.

## Resultado de la verificación (`verify_citations`)

| Categoría | Cuenta | Significado |
| --- | --- | --- |
| **grounded** | 4 | Tareas con `grounded=true` que citan chunk ids presentes en el contexto recuperado |
| **dangling** | 0 | Sin chunk ids inventados tras el reintento correctivo |
| **insufficient** | 2 | Tareas marcadas explícitamente como `grounded=false` (sin línea histórica comparable) |

## Línea grounded de ejemplo

| Módulo | Tarea | chunk_id | document_id | evidence (verbatim) |
| --- | --- | --- | --- | --- |
| Payments & Billing | Stripe checkout integration | 12 | S07-ECO-001 | `estimated_hours: 140` |

## Política aplicada

1. La primera pasada de generación produjo una citación colgante (`chunk_id=88`, ausente del contexto).
2. El reintento correctivo con un prompt endurecido reparó la citación.
3. No hizo falta degradar (no se disparó `citations_unrepaired`).

## Límites (honestos)

`verify_citations` confirma la **integridad referencial** (el chunk citado estuvo en el contexto). **No** verifica que el span de evidencia respalde *semánticamente* la cifra de engineer-days — ese hueco lo cubre el `faithfulness` de RAGAS (ver `ragas-metrics-note.md`).
