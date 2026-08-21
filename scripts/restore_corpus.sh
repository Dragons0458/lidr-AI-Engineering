#!/usr/bin/env bash
# Restore a corpus dump into the running Postgres. Goes against the database,
# never the ingest API (that would re-pay embeddings and requires X-API-Key).
set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
SERVICE="${SERVICE:-postgres}"
VECTOR_DB_USER="${VECTOR_DB_USER:-estimator}"
VECTOR_DB_NAME="${VECTOR_DB_NAME:-estimator}"

DUMP="${1:-}"
if [[ -z "${DUMP}" || ! -f "${DUMP}" ]]; then
  echo "Usage: $0 <corpus.dump>" >&2
  exit 2
fi

if ! docker compose -f "${COMPOSE_FILE}" exec -T "${SERVICE}" true >/dev/null 2>&1; then
  echo "ERROR: service '${SERVICE}' is not running (compose file: ${COMPOSE_FILE})." >&2
  exit 1
fi

echo "==> Ensuring pgvector extension exists (columns of type vector need it)"
docker compose -f "${COMPOSE_FILE}" exec -T "${SERVICE}" \
  psql -U "${VECTOR_DB_USER}" -d "${VECTOR_DB_NAME}" \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "==> Restoring ${DUMP}"
# pg_restore returns non-zero on benign notices (e.g. DROP IF EXISTS misses).
set +e
docker compose -f "${COMPOSE_FILE}" exec -T "${SERVICE}" \
  pg_restore --clean --if-exists --no-owner --no-privileges \
  -U "${VECTOR_DB_USER}" -d "${VECTOR_DB_NAME}" < "${DUMP}"
RESTORE_STATUS=$?
set -e
if [[ "${RESTORE_STATUS}" -ne 0 ]]; then
  echo "WARN: pg_restore exited ${RESTORE_STATUS} (often harmless notices). Verifying counts…" >&2
fi

echo "==> Counts"
docker compose -f "${COMPOSE_FILE}" exec -T "${SERVICE}" \
  psql -U "${VECTOR_DB_USER}" -d "${VECTOR_DB_NAME}" -c \
  "SELECT 'documents' AS t, count(*) FROM documents UNION ALL SELECT 'budget_chunks', count(*) FROM budget_chunks;"

echo
echo "If HNSW indexes are missing after restore, rebuild them"
echo "(migration 0005_session11_hnsw_multi_index) — that is the RAM spike the 2G swap is for."
echo "Done."
