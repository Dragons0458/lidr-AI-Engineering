# Runbook — ai-service no responde

1. **Síntoma:** Streamlit muestra error de conexión / 502 / 503 / timeouts hacia la API.
2. `docker compose -f docker-compose.yml ps`
3. `docker compose -f docker-compose.yml exec web python -c "import httpx;print(httpx.get('http://ai-service:8000/health',timeout=3).status_code)"`
4. `docker compose -f docker-compose.yml exec web python -c "import httpx;r=httpx.get('http://ai-service:8000/health/ready',timeout=3);print(r.status_code,r.text)"`
5. `docker compose -f docker-compose.yml logs ai-service --tail=100`
6. Causas frecuentes:
   - `migrate` falló → `logs migrate --tail=50`
   - Postgres/Redis unhealthy → ver runbook postgres
   - Embedder / índice / cliente OpenAI ausente → HTTP 503 (`Embedding service is not available`, etc.)
   - Falta `OPENAI_API_KEY` → arranque de Settings falla
   - Grafo LangGraph no compiló → endpoint 503, resto puede vivir
   - Token mismatch → 401 en todo. Comparar **por hash**, nunca `echo`:
     `printenv AI_SERVICE_TOKEN | sha256sum` vs `printenv ESTIMATE_API_KEY | sha256sum`
7. Escalera de reinicio (de menos a más destructivo):
   1. `docker compose -f docker-compose.yml restart ai-service`
   2. `docker compose -f docker-compose.yml up -d --force-recreate ai-service`
   3. Rebuild de la imagen (`docker compose -f docker-compose.yml build ai-service`)
   4. `docker compose -f docker-compose.yml down && docker compose -f docker-compose.yml up -d` (sin `-v`)
8. Si sigue: capturar logs + `X-Request-ID` de la petición fallida.

## Rama EC2 (kit cloud, si el stack corre bajo systemd)

```bash
sudo journalctl -u estimador-cag -n 200
sudo systemctl restart estimador-cag
sudo systemctl status estimador-cag
```

La unidad usa `COMPOSE_PROJECT_NAME=estimador-cag`. Si el stack se lanzó
alguna vez desde otra ruta, los volúmenes no coinciden y el corpus “desaparece”.
