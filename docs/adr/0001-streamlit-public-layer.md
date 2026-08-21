# ADR-001 — Streamlit is the public layer, not “just a frontend”

## Status

Accepted (Session 15)

## Context

The course exercise maps a Rails business backend (public) + FastAPI AI service
(private). This repo is Python-only: FastAPI + Streamlit. Streamlit already owns
user history (SQLite), wizard state, and HTTP calls into the AI service.

## Decision

Treat **`web` (Streamlit) as the single public surface**. Do **not** split it
into a separate static frontend + business API.

## Consequences

- One published host port (`8501`) in local prod-like Compose.
- Splitting Streamlit would pay the full service-boundary tax (two images,
  two healthchecks, two deploy units) without buying independent scale or a
  different language/runtime.
- Debt accepted: Streamlit still imports `app.domain.schemas` (code boundary is
  softer than the network boundary).

## Revision (Session 15 live — cloud)

In production Compose (`docker-compose.prod.yml`) the internet-facing process
is **Caddy** (`:80` / `:443`). Streamlit stays on the internal network
(`web.ports: !reset []`) and must sit behind Caddy `basic_auth` because it
has no login of its own. Localhost without basic_auth remains acceptable.
See [`../decisions.md`](../decisions.md) §8.
