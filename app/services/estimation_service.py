from collections.abc import Iterator
from dataclasses import dataclass
import time
from typing import Literal, Union

import structlog
from litellm import completion
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse

from app.config import get_settings
from app.errors.llm_error import LLMServiceError
from app.prompts.loader import (
    render_estimation_prompt,
    render_requirements_extraction_prompt,
)
from app.schemas.estimation import EstimationRequest, PreprocessingMode, TokenUsage
from app.services.sessions import ProjectMetadata

settings = get_settings()
log = structlog.get_logger()

MAX_TOKENS = 4000


@dataclass(frozen=True)
class EstimationGenerationResult:
    response: ModelResponse
    model: str
    latency_ms: int
    preprocessing: PreprocessingMode = "none"
    extracted_requirements: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    preprocessing_input_tokens: int = 0
    preprocessing_output_tokens: int = 0
    finish_reason: str | None = None


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
        response = _call_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            thinking_budget=None,
        )
    except Exception as e:
        log.error(
            "requirements_extraction_failed",
            error=str(e),
            provider=settings.LLM_PROVIDER,
        )
        raise LLMServiceError(f"Requirements extraction failed: {e}") from e

    input_tokens, output_tokens = _response_usage(response)
    extracted_requirements = _response_content(response).strip()
    return extracted_requirements, TokenUsage(
        tokens_used=input_tokens + output_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def generate_estimation(
    request: EstimationRequest,
    prompt_version: Literal["v1", "v2"] = "v1",
    project_metadata: ProjectMetadata | None = None,
    messages: list[dict[str, str]] | None = None,
) -> EstimationGenerationResult:
    """Generate an estimation for a software development project based on a meeting summary."""
    started_at = time.perf_counter()
    model = request.model or settings.LLM_MODEL
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
        provider=settings.LLM_PROVIDER,
        model=model,
        project_type=effective_request.project_type.value,
        detail_level=effective_request.detail_level.value,
        output_format=effective_request.output_format.value,
        prompt_version=prompt_version,
        preprocessing=request.preprocessing,
    )

    try:
        response = _call_completion(
            model=model,
            messages=messages,
            max_tokens=request.max_tokens,
            thinking_budget=request.thinking_budget,
        )
        main_input_tokens, main_output_tokens = _response_usage(response)
        prep_in = preprocessing_usage.input_tokens or 0
        prep_out = preprocessing_usage.output_tokens or 0
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        log.info(
            "llm_response_received",
            provider=settings.LLM_PROVIDER,
            model=model,
            input_tokens=main_input_tokens,
            output_tokens=main_output_tokens,
            preprocessing_input_tokens=prep_in,
            preprocessing_output_tokens=prep_out,
            latency_ms=latency_ms,
        )
        return EstimationGenerationResult(
            response=response,
            model=model,
            latency_ms=latency_ms,
            preprocessing=request.preprocessing,
            extracted_requirements=extracted_requirements,
            input_tokens=main_input_tokens,
            output_tokens=main_output_tokens,
            preprocessing_input_tokens=prep_in,
            preprocessing_output_tokens=prep_out,
            finish_reason=_finish_reason(response),
        )
    except Exception as e:
        log.error("llm_call_failed", error=str(e), provider=settings.LLM_PROVIDER)
        raise LLMServiceError(f"LLM call failed: {e}") from e


def generate_estimation_stream(
    request: EstimationRequest,
    prompt_version: Literal["v1", "v2"] = "v1",
    project_metadata: ProjectMetadata | None = None,
) -> Iterator[str]:
    """Generate an estimation stream for a software project summary."""
    model = request.model or settings.LLM_MODEL
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
        provider=settings.LLM_PROVIDER,
        model=model,
        project_type=effective_request.project_type.value,
        detail_level=effective_request.detail_level.value,
        output_format=effective_request.output_format.value,
        prompt_version=prompt_version,
        preprocessing=request.preprocessing,
    )

    try:
        stream = _call_completion(
            model=model,
            messages=messages,
            max_tokens=request.max_tokens,
            thinking_budget=request.thinking_budget,
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        log.error(
            "llm_stream_call_failed", error=str(e), provider=settings.LLM_PROVIDER
        )
        raise LLMServiceError(f"LLM stream call failed: {e}") from e


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


def _call_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    thinking_budget: int | None = None,
    stream: bool = False,
):
    call_kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if stream:
        call_kwargs["stream"] = True

    if thinking_budget is not None and settings.LLM_PROVIDER == "anthropic":
        call_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
        call_kwargs["max_tokens"] = max(max_tokens, thinking_budget + 1024)
    else:
        if thinking_budget is not None:
            log.warning(
                "thinking_budget_ignored_for_provider",
                provider=settings.LLM_PROVIDER,
                model=model,
            )
        call_kwargs["reasoning_effort"] = "none"

    return completion(**call_kwargs)


def _response_usage(
    response: Union[ModelResponse, CustomStreamWrapper],
) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


def _response_content(response: ModelResponse) -> str:
    return response.choices[0].message.content or ""


def _finish_reason(response: ModelResponse) -> str | None:
    return getattr(response.choices[0], "finish_reason", None)
