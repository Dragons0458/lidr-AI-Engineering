#!/usr/bin/env bash
# Dump the pgvector corpus (already-paid embeddings) to a custom-format archive.
# Restore with scripts/restore_corpus.sh — do NOT re-ingest through the API.
set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
SERVICE="${SERVICE:-postgres}"
VECTOR_DB_USER="${VECTOR_DB_USER:-estimator}"
VECTOR_DB_NAME="${VECTOR_DB_NAME:-estimator}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${1:-${ROOT}/backups/corpus-${STAMP}.dump}"

if ! docker compose -f "${COMPOSE_FILE}" exec -T "${SERVICE}" true >/dev/null 2>&1; then
  echo "ERROR: service '${SERVICE}' is not running (compose file: ${COMPOSE_FILE})." >&2
  echo "Start the stack first: docker compose -f ${COMPOSE_FILE} up -d" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT}")"

echo "==> Dumping ${VECTOR_DB_NAME} from ${SERVICE} → ${OUT}"
docker compose -f "${COMPOSE_FILE}" exec -T "${SERVICE}" \
  pg_dump -Fc --no-owner --no-privileges -U "${VECTOR_DB_USER}" "${VECTOR_DB_NAME}" \
  > "${OUT}"

SIZE="$(du -h "${OUT}" | awk '{print $1}')"
echo "Wrote ${OUT} (${SIZE})"
echo
echo "Copy to the target host:"
echo "  scp ${OUT} ubuntu@<host>:/opt/estimador-cag/backups/"
echo
echo "Restore:"
echo "  ./scripts/restore_corpus.sh ${OUT}"
