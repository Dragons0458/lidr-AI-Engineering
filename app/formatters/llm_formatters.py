from datetime import datetime
from typing import Union

from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse

from app.config import get_settings
from app.schemas.estimation import EstimationResponse, TokenUsage
from app.services.calc_service import calculate_cost
from app.services.estimation_service import EstimationGenerationResult

settings = get_settings()


def format_response(
    response: Union[EstimationGenerationResult, ModelResponse, CustomStreamWrapper],
    prompt_version: str = "v1",
) -> EstimationResponse:
    """Format the LLM response into an EstimationResponse object."""
    generation_result = (
        response if isinstance(response, EstimationGenerationResult) else None
    )
    model_response = generation_result.response if generation_result else response
    model = generation_result.model if generation_result else settings.LLM_MODEL
    main_input_tokens = (
        generation_result.input_tokens
        if generation_result and generation_result.input_tokens is not None
        else model_response.usage.prompt_tokens
    )
    main_output_tokens = (
        generation_result.output_tokens
        if generation_result and generation_result.output_tokens is not None
        else model_response.usage.completion_tokens
    )
    prep_in = generation_result.preprocessing_input_tokens if generation_result else 0
    prep_out = generation_result.preprocessing_output_tokens if generation_result else 0
    finish_reason = (
        generation_result.finish_reason
        if generation_result
        else getattr(model_response.choices[0], "finish_reason", None)
    )

    cost = calculate_cost(
        model,
        main_input_tokens + prep_in,
        main_output_tokens + prep_out,
    )

    return EstimationResponse(
        estimation=model_response.choices[0].message.content or "",
        timestamp=datetime.now(),
        model=model,
        provider=settings.LLM_PROVIDER,
        prompt_version=prompt_version,
        usage=TokenUsage(
            cost_estimate=cost.get("total") if cost else None,
            tokens_used=main_input_tokens + main_output_tokens + prep_in + prep_out,
            input_tokens=main_input_tokens,
            output_tokens=main_output_tokens,
            preprocessing_input_tokens=prep_in,
            preprocessing_output_tokens=prep_out,
        ),
        latency_ms=generation_result.latency_ms if generation_result else 0,
        finish_reason=finish_reason,
        preprocessing=generation_result.preprocessing if generation_result else "none",
        extracted_requirements=(
            generation_result.extracted_requirements if generation_result else None
        ),
    )
