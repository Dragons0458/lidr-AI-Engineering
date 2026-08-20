# ADR-002 — One Postgres for relational and vector data

## Status

Accepted (Session 15)

## Context

The exercise reference uses a relational DB plus a separate vector DB (Qdrant).
This project has used **pgvector inside Postgres** since Session 8. Alembic
tables, LangGraph checkpoints, and the chunk corpus share `DATABASE_URL`.

## Decision

Keep **one** `postgres` service (`pgvector/pgvector:pg16`). Do not introduce
Qdrant or a second Postgres for local/prod-like Compose.

## Consequences

- Simpler DSN, one volume, one healthcheck.
- Independent scale of the vector index is deferred until it actually hurts
  (then reconsider a dedicated vector store).
