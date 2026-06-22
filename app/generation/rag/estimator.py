"""End-to-end RAG estimation orchestrator (Session 9)."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import structlog

from app.config import get_settings
from app.generation.rag.context_assembler import (
    build_context_block,
    truncate_to_token_budget,
)
from app.generation.rag.errors import GenerationError, MalformedEstimateError
from app.generation.rag.observability import log_stage
from app.generation.rag.prompt_builder import build_system_prompt, build_user_message
from app.generation.rag.query_reformulator import compose_search_text, reformulate_query
from app.generation.rag.retriever import search_chunks
from app.generation.rag.schemas import Estimate, EstimationQuery
from app.generation.rag.validation import check_coherence, validate_citations

log = structlog.get_logger()

_KNOWN_SECTORS = {"finance", "ecommerce", "healthcare", "industrial"}


async def generate_estimate(
    context_block: str,
    structured_query: EstimationQuery,
) -> Estimate:
    """Generate a grounded :class:`Estimate` from an assembled context block."""
    return await _generate(context_block, structured_query)


async def _generate(
    context_block: str,
    structured_query: EstimationQuery,
    *,
    feedback: str | None = None,
) -> Estimate:
    """Single generation call. ``feedback`` appends a correction note for retries."""
    from app.dependencies import get_llm_wrapper

    settings = get_settings()
    wrapper = get_llm_wrapper()

    user_message = build_user_message(context_block, structured_query)
    if feedback:
        user_message += f"\n\n<correction>\n{feedback}\n</correction>"

    from app.foundation.llm.wrapper import is_reasoning_model

    structured_kwargs: dict = {
        "system_prompt": build_system_prompt(),
        "user_message": user_message,
        "response_model": Estimate,
        "model_override": settings.GENERATION_MODEL,
        "max_tokens": settings.GENERATION_MAX_TOKENS,
    }
    if is_reasoning_model(settings.GENERATION_MODEL):
        structured_kwargs["reasoning_effort"] = settings.GENERATION_REASONING_EFFORT

    try:
        estimate, _meta = await asyncio.to_thread(
            wrapper.complete_structured,
            **structured_kwargs,
        )
        return estimate
    except Exception as exc:  # noqa: BLE001
        raise GenerationError("Grounded estimate generation failed.") from exc


def _insufficient(explanation: str) -> Estimate:
    """Build the canonical insufficient-context estimate (no numbers)."""
    return Estimate(
        total_engineer_days=None,
        duration_weeks=None,
        confidence="insufficient",
        reasoning="Retrieval did not surface enough relevant historical budgets.",
        insufficient_context_explanation=explanation,
    )


def _current_request_id() -> str:
    """Reuse the HTTP request id bound by the middleware, or mint one."""
    bound = structlog.contextvars.get_contextvars().get("request_id")
    return bound or str(uuid4())


async def estimate_from_transcript(
    transcript: str,
    idempotency_key: str | None = None,
) -> Estimate:
    """Run the full transcript → grounded estimate pipeline."""
    from app.dependencies import get_embedder, get_idempotency_store, get_token_encoder

    settings = get_settings()
    request_id = _current_request_id()
    store = get_idempotency_store()

    if idempotency_key:
        cached = await asyncio.to_thread(store.get, idempotency_key)
        if cached is not None:
            log.info(
                "idempotency_hit",
                request_id=request_id,
                idempotency_key=idempotency_key,
            )
            return cached

    with log_stage("reformulation", request_id):
        query = await reformulate_query(transcript)

    with log_stage("embedding", request_id):
        search_text = compose_search_text(query)
        embedder = get_embedder()
        if embedder is None:
            raise GenerationError("Embedding service is not available (no OpenAI key).")
        query_embedding = await asyncio.to_thread(embedder.embed_one, search_text)

    sector = query.sector.lower().strip() if query.sector else None
    sectors = [sector] if sector in _KNOWN_SECTORS else None
    with log_stage("retrieval", request_id, sectors=sectors):
        retrieval = await search_chunks(
            query_embedding,
            query_text=search_text,
            top_k=settings.RERANK_TOP_N
            if settings.RERANKER_ENABLED
            else settings.RETRIEVAL_TOP_K,
            distance_threshold=settings.RETRIEVAL_DISTANCE_THRESHOLD,
            sectors=sectors,
        )

    if retrieval.low_confidence:
        log.info(
            "retrieval_soft_fail",
            request_id=request_id,
            candidates=retrieval.candidates_evaluated,
        )
        estimate = _insufficient(
            "No historical budgets crossed the relevance threshold for this project."
        )
        if idempotency_key:
            await asyncio.to_thread(store.set, idempotency_key, estimate)
        return estimate

    encoder = get_token_encoder()
    kept = truncate_to_token_budget(
        retrieval.chunks, settings.MAX_CONTEXT_TOKENS, encoder
    )
    context_block = build_context_block(kept)

    with log_stage("generation", request_id, sources=len(kept)):
        estimate = await generate_estimate(context_block, structured_query=query)

    fabricated = validate_citations(estimate, kept)
    if fabricated:
        feedback = (
            f"your previous response cited invalid source ids: {fabricated}. "
            "Only cite ids that appear in the <sources> block."
        )
        with log_stage("citation_retry", request_id, fabricated=fabricated):
            estimate = await _generate(context_block, query, feedback=feedback)
        if validate_citations(estimate, kept):
            log.warning("citations_unrepaired", request_id=request_id)
            estimate = estimate.model_copy(update={"confidence": "low"})

    if not check_coherence(estimate):
        feedback = (
            'when confidence is "insufficient", total_engineer_days and '
            "duration_weeks must be null, modules must be empty and "
            "insufficient_context_explanation must be filled; otherwise provide "
            "the modules, tasks and numbers."
        )
        with log_stage("coherence_repair", request_id):
            estimate = await _generate(context_block, query, feedback=feedback)
        if not check_coherence(estimate):
            raise MalformedEstimateError(
                "Estimate violates the insufficient-context coherence rule."
            )

    if idempotency_key:
        await asyncio.to_thread(store.set, idempotency_key, estimate)

    return estimate
