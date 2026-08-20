# ADR-004 — Streamlit SQLite on a named volume

## Status

Accepted (Session 15)

## Context

`streamlit_ui/store.py` persists UI history in SQLite. Rewriting it to Postgres
is out of scope for a dockerisation exercise. Bind-mounting the repo path fails
in prod-like runs (no host tree inside the image).

## Decision

Mount named volume `streamlit_data` at `/app/streamlit_ui/data` and set
`STREAMLIT_DB_PATH=/app/streamlit_ui/data/frontend.db`.

## Consequences

- `docker compose down` (without `-v`) keeps estimation history.
- `down -v` deliberately wipes it — teach the difference.
