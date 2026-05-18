from collections.abc import Iterator
from typing import Literal, Union

import structlog
from litellm import completion
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse

from app.config import get_settings
from app.errors.llm_error import LLMServiceError
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import EstimationRequest
from app.services.sessions import ProjectMetadata

settings = get_settings()
log = structlog.get_logger()

MAX_TOKENS = 4000


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


def generate_estimation(
    request: EstimationRequest,
    prompt_version: Literal["v1", "v2"] = "v1",
    project_metadata: ProjectMetadata | None = None,
    messages: list[dict[str, str]] | None = None,
) -> Union[ModelResponse, CustomStreamWrapper]:
    """Generate an estimation for a software development project based on a meeting summary."""
    messages = messages or _build_messages(request, prompt_version, project_metadata)

    log.info(
        "generating_estimation",
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
        project_type=request.project_type.value,
        detail_level=request.detail_level.value,
        output_format=request.output_format.value,
        prompt_version=prompt_version,
    )

    try:
        response = completion(
            model=settings.LLM_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            reasoning_effort="none",
        )

        log.info(
            "llm_response_received",
            provider=settings.LLM_PROVIDER,
            model=settings.LLM_MODEL,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        return response
    except Exception as e:
        log.error("llm_call_failed", error=str(e), provider=settings.LLM_PROVIDER)
        raise LLMServiceError(f"LLM call failed: {e}") from e


def generate_estimation_stream(
    request: EstimationRequest,
    prompt_version: Literal["v1", "v2"] = "v1",
    project_metadata: ProjectMetadata | None = None,
) -> Iterator[str]:
    """Generate an estimation stream for a software project summary."""
    messages = _build_messages(request, prompt_version, project_metadata)

    log.info(
        "generating_estimation_stream",
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
        project_type=request.project_type.value,
        detail_level=request.detail_level.value,
        output_format=request.output_format.value,
        prompt_version=prompt_version,
    )

    try:
        stream = completion(
            model=settings.LLM_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            reasoning_effort="none",
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
