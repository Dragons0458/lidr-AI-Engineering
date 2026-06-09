from datetime import datetime

from app.domain.schemas.estimation import EstimationResponse, TokenUsage
from app.foundation.llm.pricing import calculate_cost
from app.domain.estimation_service import EstimationGenerationResult


def format_response(
    response: EstimationGenerationResult,
    prompt_version: str = "v1",
) -> EstimationResponse:
    """Format the LLM generation result into an EstimationResponse."""
    main_input_tokens = response.input_tokens or 0
    main_output_tokens = response.output_tokens or 0
    prep_in = response.preprocessing_input_tokens
    prep_out = response.preprocessing_output_tokens

    cost = calculate_cost(
        response.model,
        main_input_tokens + prep_in,
        main_output_tokens + prep_out,
    )
    cost_usd = response.cost_usd or (float(cost["total"]) if cost else 0.0)

    return EstimationResponse(
        estimation=response.estimation,
        timestamp=datetime.now(),
        model=response.model,
        provider=response.provider,
        prompt_version=prompt_version,
        usage=TokenUsage(
            cost_estimate=cost.get("total") if cost else cost_usd,
            tokens_used=main_input_tokens + main_output_tokens + prep_in + prep_out,
            input_tokens=main_input_tokens,
            output_tokens=main_output_tokens,
            preprocessing_input_tokens=prep_in,
            preprocessing_output_tokens=prep_out,
        ),
        latency_ms=response.latency_ms,
        finish_reason=response.finish_reason,
        preprocessing=response.preprocessing,
        extracted_requirements=response.extracted_requirements,
        cache_hit=response.cache_hit,
        cost_usd=cost_usd,
        out_of_scope=response.out_of_scope,
    )
