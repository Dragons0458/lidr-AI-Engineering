# Runbook — Rotar clave LLM

1. Generar nueva clave en el proveedor (OpenAI / Anthropic / Google).
2. Actualizar `.env` → `OPENAI_API_KEY` (u otra según `LLM_PROVIDER`).
3. **No** rebuild de imagen: los secretos no van en capas.
4. `docker compose -f docker-compose.yml up -d --force-recreate ai-service`
5. Verificar: estimación corta desde la UI o `scripts/smoke_test_s15.py`.
6. Revocar la clave antigua en el panel del proveedor.
7. Si también rotas el token de servicio: actualizar `ESTIMATE_API_KEY` /
   `AI_SERVICE_TOKEN` (+ `RETRIEVAL_API_KEY` si aplica) y recrear `web` +
   `ai-service`.
