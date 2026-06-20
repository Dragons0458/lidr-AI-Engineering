"""Query understanding: turn a raw meeting transcript into a structured brief."""

from __future__ import annotations

import asyncio

import structlog

from app.config import get_settings
from app.generation.rag.errors import ReformulationError
from app.generation.rag.schemas import EstimationQuery

log = structlog.get_logger()

_SYSTEM_PROMPT = (
    "You are a software-delivery analyst. Extract a structured project brief from "
    "a raw, messy client meeting transcript. Capture ONLY what the client wants to "
    "build and the constraints that bound it; ignore small talk, anecdotes and "
    "digressions. Normalise everything to concise technical English regardless of "
    "the transcript language. Leave a field empty/unknown when the transcript gives "
    "no evidence for it — never invent technologies, sectors or regulations."
)

_FALLBACK_SYSTEM_PROMPT = (
    "Rewrite the following client meeting transcript as a single short technical "
    "search query in English describing the software to build. One sentence, no "
    "preamble."
)


def _structured_models(settings, wrapper) -> list[str]:
    """Models to try for structured extraction (deduped, REFORMULATION then PRIMARY)."""
    primary = getattr(wrapper, "primary_model", settings.REFORMULATION_MODEL)
    candidates = [settings.REFORMULATION_MODEL, primary]
    seen: set[str] = set()
    ordered: list[str] = []
    for model in candidates:
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered


async def reformulate_query(transcript: str) -> EstimationQuery:
    """Distill a transcript into a structured :class:`EstimationQuery`."""
    from app.dependencies import get_llm_wrapper

    settings = get_settings()
    wrapper = get_llm_wrapper()
    last_exc: Exception | None = None

    for model in _structured_models(settings, wrapper):
        try:
            query, _meta = await asyncio.to_thread(
                wrapper.complete_structured,
                system_prompt=_SYSTEM_PROMPT,
                user_message=transcript,
                response_model=EstimationQuery,
                model_override=model,
                max_tokens=settings.GENERATION_MAX_TOKENS,
            )
            return query
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning(
                "reformulation_fallback",
                reason="structured_extraction_failed",
                model=model,
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )

    fallback_model = getattr(wrapper, "primary_model", settings.REFORMULATION_MODEL)
    try:
        result = await asyncio.to_thread(
            wrapper.complete,
            messages=[
                {"role": "system", "content": _FALLBACK_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            model=fallback_model,
            max_tokens=settings.GENERATION_MAX_TOKENS,
        )
        rewritten = (result.get("estimation") or "").strip()
        if not rewritten:
            raise ReformulationError("Fallback rewrite returned an empty query.")
        return EstimationQuery(function=rewritten)
    except ReformulationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ReformulationError("Query reformulation failed.") from (last_exc or exc)


def compose_search_text(query: EstimationQuery) -> str:
    """Compose the short technical string to embed from an :class:`EstimationQuery`."""
    parts: list[str] = [query.function.strip()] if query.function.strip() else []

    if query.technologies:
        parts.append("with " + ", ".join(query.technologies))
    if query.sector:
        parts.append(f"for {query.sector}")
    if query.country:
        parts.append(f"in {query.country}")
    if query.regulations:
        parts.append(", ".join(query.regulations) + "-compliant")

    return " ".join(parts).strip()
