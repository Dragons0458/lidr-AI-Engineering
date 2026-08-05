# Runbook — Postgres caído

1. **Síntoma:** `/health` = 200, `/health/ready` = 503 con `postgres` en error; UI 503.
2. `docker compose -f docker-compose.yml ps postgres`
3. `docker compose -f docker-compose.yml logs postgres --tail=100`
4. `docker compose -f docker-compose.yml exec postgres pg_isready -U ${POSTGRES_USER:-estimator} -d ${POSTGRES_DB:-estimator}`
5. Causas frecuentes:
   - Volumen corrupto tras kill brusco
   - Credenciales `POSTGRES_*` distintas de `DATABASE_URL`
   - Disco host lleno
6. `docker compose -f docker-compose.yml restart postgres`
7. Esperar `healthy`, luego `restart ai-service`
8. **No** usar `down -v` salvo aceptar pérdida del corpus.
