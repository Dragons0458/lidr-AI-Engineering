# Runbook — ai-service no responde

1. **Síntoma:** Streamlit muestra error de conexión / 502 / timeouts hacia la API.
2. `docker compose -f docker-compose.yml ps`
3. `docker compose -f docker-compose.yml exec web python -c "import httpx;print(httpx.get('http://ai-service:8000/health',timeout=3).status_code)"`
4. `docker compose -f docker-compose.yml exec web python -c "import httpx;r=httpx.get('http://ai-service:8000/health/ready',timeout=3);print(r.status_code,r.text)"`
5. `docker compose -f docker-compose.yml logs ai-service --tail=100`
6. Causas frecuentes:
   - `migrate` falló → `logs migrate --tail=50`
   - Postgres/Redis unhealthy → ver runbook postgres
   - Falta `OPENAI_API_KEY` → arranque de Settings falla
   - Grafo LangGraph no compiló → endpoint 503, resto puede vivir
7. `docker compose -f docker-compose.yml restart ai-service`
8. Si sigue: `docker compose -f docker-compose.yml up -d --force-recreate ai-service`
9. Escalar: capturar logs + `X-Request-ID` de la petición fallida.
