from datetime import datetime
from typing import Union

from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse

from app.config import get_settings
from app.schemas.estimation import EstimationResponse, TokenUsage
from app.services.calc_service import calculate_cost

settings = get_settings()


def format_response(
    response: Union[ModelResponse, CustomStreamWrapper],
) -> EstimationResponse:
    """Format the LLM response into an EstimationResponse object."""

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = calculate_cost(settings.LLM_MODEL, input_tokens, output_tokens)

    return EstimationResponse(
        estimation=response.choices[0].message.content or "",
        timestamp=datetime.now(),
        model=settings.LLM_MODEL,
        provider=settings.LLM_PROVIDER,
        usage=TokenUsage(
            cost_estimate=cost.get("total"), tokens_used=input_tokens + output_tokens
        ),
    )
