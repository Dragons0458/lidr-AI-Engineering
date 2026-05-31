from collections.abc import Iterator
from dataclasses import dataclass
import time
from typing import Literal

import structlog

from app.config import get_settings
from app.dependencies import get_cache, get_llm_wrapper, get_semantic_cache
from app.errors.llm_error import LLMServiceError
from app.guardrails.input import check_input
from app.guardrails.output import enforce_output
from app.prompts.loader import (
    render_estimation_prompt,
    render_requirements_extraction_prompt,
)
from app.schemas.estimation import EstimationRequest, PreprocessingMode, TokenUsage
from app.services.cache import EstimationCache
from app.services.sessions import ProjectMetadata

settings = get_settings()
log = structlog.get_logger()


@dataclass(frozen=True)
class EstimationGenerationResult:
    estimation: str
    model: str
    provider: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    cache_hit: bool = False
    cost_usd: float = 0.0
    preprocessing: PreprocessingMode = "none"
    extracted_requirements: str | None = None
    preprocessing_input_tokens: int = 0
    preprocessing_output_tokens: int = 0
    out_of_scope: bool = False


def _attachments_text(request: EstimationRequest) -> list[str]:
    return [attachment.content for attachment in request.attachments or []]


def _run_input_guardrails(request: EstimationRequest) -> None:
    if not settings.INPUT_GUARDRAILS_ENABLED:
        return
    check_input(
        request.description,
        attachments_text=_attachments_text(request),
    )


def _build_messages(
    request: EstimationRequest,
    prompt_version: Literal["v1", "v2"],
    project_metadata: ProjectMetadata | None = None,
) -> list[dict[str, str]]:
    """Build LLM chat messages from Jinja templates."""
    system_prompt, user_prompt = render_estimation_prompt(
        request, version=prompt_version, project_metadata=project_metadata
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_requirements(
    request: EstimationRequest, *, model: str, max_tokens: int
) -> tuple[str, TokenUsage]:
    """Extract cleaned requirements in a first, cheap LLM pass."""
    system_prompt, user_prompt = render_requirements_extraction_prompt(request)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        result = get_llm_wrapper().complete(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            thinking_budget=None,
            use_cache=True,
        )
    except Exception as e:
        log.error("requirements_extraction_failed", error=str(e), model=model)
        raise LLMServiceError(f"Requirements extraction failed: {e}") from e

    usage = result["usage"]
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    extracted_requirements = result["estimation"].strip()
    return extracted_requirements, TokenUsage(
        tokens_used=input_tokens + output_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _result_from_cached(
    cached: dict,
    *,
    latency_ms: int,
    preprocessing: PreprocessingMode,
    extracted_requirements: str | None,
    preprocessing_usage: TokenUsage,
) -> EstimationGenerationResult:
    usage = cached.get("usage") or {}
    prep_in = preprocessing_usage.input_tokens or 0
    prep_out = preprocessing_usage.output_tokens or 0
    return EstimationGenerationResult(
        estimation=cached.get("estimation", ""),
        model=cached.get("model", settings.PRIMARY_MODEL),
        provider=cached.get("provider", "unknown"),
        latency_ms=latency_ms,
        preprocessing=preprocessing,
        extracted_requirements=extracted_requirements,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        preprocessing_input_tokens=prep_in,
        preprocessing_output_tokens=prep_out,
        finish_reason=cached.get("finish_reason"),
        cache_hit=True,
        cost_usd=float(cached.get("cost_usd", 0.0)),
    )


def _apply_output_guardrail(
    result: EstimationGenerationResult,
) -> EstimationGenerationResult:
    if not settings.OUTPUT_GUARDRAILS_ENABLED:
        return result

    guarded = enforce_output(result.estimation)
    return EstimationGenerationResult(
        estimation=guarded.text,
        model=result.model,
        provider=result.provider,
        latency_ms=result.latency_ms,
        preprocessing=result.preprocessing,
        extracted_requirements=result.extracted_requirements,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        preprocessing_input_tokens=result.preprocessing_input_tokens,
        preprocessing_output_tokens=result.preprocessing_output_tokens,
        finish_reason=result.finish_reason,
        cache_hit=result.cache_hit,
        cost_usd=result.cost_usd,
        out_of_scope=guarded.out_of_scope,
    )


def _semantic_payload(result: EstimationGenerationResult) -> dict:
    return {
        "estimation": result.estimation,
        "model": result.model,
        "provider": result.provider,
        "usage": {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        },
        "cost_usd": result.cost_usd,
        "finish_reason": result.finish_reason,
    }


def generate_estimation(
    request: EstimationRequest,
    prompt_version: Literal["v1", "v2"] = "v1",
    project_metadata: ProjectMetadata | None = None,
    messages: list[dict[str, str]] | None = None,
    *,
    use_cache: bool = True,
) -> EstimationGenerationResult:
    """Generate an estimation for a software development project."""
    _run_input_guardrails(request)

    started_at = time.perf_counter()
    model = request.model or settings.PRIMARY_MODEL
    extracted_requirements: str | None = None
    preprocessing_usage = TokenUsage(tokens_used=0, input_tokens=0, output_tokens=0)
    effective_request = request

    if request.preprocessing == "two_phase":
        extracted_requirements, preprocessing_usage = extract_requirements(
            request, model=model, max_tokens=request.max_tokens
        )
        if extracted_requirements:
            effective_request = request.model_copy(
                update={"description": extracted_requirements}
            )

    messages = _resolve_messages(
        effective_request,
        prompt_version=prompt_version,
        project_metadata=project_metadata,
        existing_messages=messages,
    )

    log.info(
        "generating_estimation",
        model=model,
        project_type=effective_request.project_type.value,
        detail_level=effective_request.detail_level.value,
        output_format=effective_request.output_format.value,
        prompt_version=prompt_version,
        preprocessing=request.preprocessing,
        use_cache=use_cache,
    )

    cache_key = EstimationCache.make_key(
        messages=messages,
        model=model,
        max_tokens=request.max_tokens,
        thinking_budget=request.thinking_budget,
    )

    if use_cache and settings.CACHE_ENABLED:
        cached = get_cache().get(cache_key)
        if cached:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            result = _apply_output_guardrail(
                _result_from_cached(
                    cached,
                    latency_ms=latency_ms,
                    preprocessing=request.preprocessing,
                    extracted_requirements=extracted_requirements,
                    preprocessing_usage=preprocessing_usage,
                )
            )
            log.info(
                "exact_cache_hit",
                latency_ms=latency_ms,
                out_of_scope=result.out_of_scope,
            )
            return result

    semantic_cache = get_semantic_cache()
    if use_cache and semantic_cache is not None:
        semantic_hit = semantic_cache.lookup(effective_request, prompt_version)
        if semantic_hit:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            result = _apply_output_guardrail(
                _result_from_cached(
                    semantic_hit,
                    latency_ms=latency_ms,
                    preprocessing=request.preprocessing,
                    extracted_requirements=extracted_requirements,
                    preprocessing_usage=preprocessing_usage,
                )
            )
            log.info(
                "semantic_cache_hit",
                latency_ms=latency_ms,
                out_of_scope=result.out_of_scope,
            )
            return result

    try:
        result_dict = get_llm_wrapper().complete(
            messages=messages,
            model=model,
            max_tokens=request.max_tokens,
            thinking_budget=request.thinking_budget,
            use_cache=use_cache,
        )
        usage = result_dict["usage"]
        prep_in = preprocessing_usage.input_tokens or 0
        prep_out = preprocessing_usage.output_tokens or 0
        main_input = usage["input_tokens"]
        main_output = usage["output_tokens"]
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        log.info(
            "llm_response_received",
            model=result_dict["model"],
            provider=result_dict["provider"],
            input_tokens=main_input,
            output_tokens=main_output,
            preprocessing_input_tokens=prep_in,
            preprocessing_output_tokens=prep_out,
            latency_ms=latency_ms,
            cache_hit=result_dict.get("cache_hit", False),
        )
        result = _apply_output_guardrail(
            EstimationGenerationResult(
                estimation=result_dict["estimation"],
                model=result_dict["model"],
                provider=result_dict["provider"],
                latency_ms=latency_ms,
                preprocessing=request.preprocessing,
                extracted_requirements=extracted_requirements,
                input_tokens=main_input,
                output_tokens=main_output,
                preprocessing_input_tokens=prep_in,
                preprocessing_output_tokens=prep_out,
                finish_reason=result_dict.get("finish_reason"),
                cache_hit=bool(result_dict.get("cache_hit", False)),
                cost_usd=float(result_dict.get("cost_usd", 0.0)),
            )
        )

        if (
            use_cache
            and semantic_cache is not None
            and not result.cache_hit
            and not result.out_of_scope
        ):
            semantic_cache.store(
                effective_request,
                _semantic_payload(result),
                prompt_version,
            )

        return result
    except Exception as e:
        log.error("llm_call_failed", error=str(e))
        raise LLMServiceError(f"LLM call failed: {e}") from e


def generate_estimation_stream(
    request: EstimationRequest,
    prompt_version: Literal["v1", "v2"] = "v1",
    project_metadata: ProjectMetadata | None = None,
    *,
    use_cache: bool = True,
) -> Iterator[str]:
    """Generate an estimation stream for a software project summary.

    Output guardrails (scope/PII) are not applied to streamed chunks; only the
    blocking ``generate_estimation`` path enforces them.
    """
    _run_input_guardrails(request)

    model = request.model or settings.PRIMARY_MODEL
    effective_request = request
    if request.preprocessing == "two_phase":
        extracted_requirements, _ = extract_requirements(
            request, model=model, max_tokens=request.max_tokens
        )
        if extracted_requirements:
            effective_request = request.model_copy(
                update={"description": extracted_requirements}
            )

    messages = _build_messages(effective_request, prompt_version, project_metadata)

    log.info(
        "generating_estimation_stream",
        model=model,
        project_type=effective_request.project_type.value,
        prompt_version=prompt_version,
        use_cache=use_cache,
    )

    semantic_cache = get_semantic_cache()
    if use_cache and semantic_cache is not None:
        semantic_hit = semantic_cache.lookup(effective_request, prompt_version)
        if semantic_hit:
            estimation = semantic_hit.get("estimation", "")
            if estimation:
                log.info("semantic_cache_stream_hit")
                yield estimation
                return

    collected: list[str] = []
    try:
        for chunk in get_llm_wrapper().complete_stream(
            messages=messages,
            model=model,
            max_tokens=request.max_tokens,
            thinking_budget=request.thinking_budget,
            use_cache=use_cache,
        ):
            collected.append(chunk)
            yield chunk
    except Exception as e:
        log.error("llm_stream_call_failed", error=str(e))
        raise LLMServiceError(f"LLM stream call failed: {e}") from e

    if use_cache and semantic_cache is not None and collected:
        rendered = "".join(collected)
        if not rendered.strip().startswith("Out of scope:"):
            semantic_cache.store(
                effective_request,
                {
                    "estimation": rendered,
                    "model": model,
                    "provider": "unknown",
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "cost_usd": 0.0,
                    "finish_reason": "stop",
                },
                prompt_version,
            )


def _resolve_messages(
    request: EstimationRequest,
    *,
    prompt_version: Literal["v1", "v2"],
    project_metadata: ProjectMetadata | None,
    existing_messages: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    if existing_messages is None:
        return _build_messages(request, prompt_version, project_metadata)

    system_prompt, user_prompt = render_estimation_prompt(
        request, version=prompt_version, project_metadata=project_metadata
    )
    resolved_messages = [message.copy() for message in existing_messages]
    if resolved_messages:
        resolved_messages[0] = {**resolved_messages[0], "content": system_prompt}
        resolved_messages[-1] = {**resolved_messages[-1], "content": user_prompt}
    return resolved_messages
