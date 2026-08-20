# ADR-003 — One Dockerfile, two targets

## Status

Accepted (Session 15)

## Context

`ai-service` and `web` share `pyproject.toml` / `uv.lock`. A second Dockerfile
would duplicate the expensive `uv sync` layer (including `torch` via
`sentence-transformers`).

## Decision

One multi-stage `Dockerfile` with targets `builder` → `base` → `ai-service` |
`web`. Compose selects `target:` per service.

## Consequences

- Artefact-per-service spirit without duplicating ~2 GB of deps.
- `web` also copies `app/` because Streamlit imports schemas/helpers (documented
  debt; network frontier still holds).
