from typing import Union

import structlog
from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES, format_examples_for_prompt
from app.errors.llm_error import LLMServiceError
from litellm import completion
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse

settings = get_settings()
log = structlog.get_logger()

MAX_TOKENS = 4000


def _generate_system_prompt() -> str:
    """Generate the system prompt for the LLM."""
    examples = format_examples_for_prompt(ESTIMATION_EXAMPLES)
    system_prompt = f"""
            You are a helpful assistant that generates estimations for software development projects.
            You will be given a summary of a meeting with a client and you will need to generate an estimation for the project.
            You will use the following examples to generate the estimation:
            {examples}

            Response structure:
            - Maximum 500 words
            - Must be in Spanish
            - Maximum 10 tasks

            Behavioral constraints:
            - Must be a helpful assistant
            - Must be concise and to the point
            - Must be accurate and realistic
            - Must be easy to understand
            - Must be easy to follow
            - Must be easy to implement
            - Must be easy to maintain
            - Must be easy to scale

            Do not:
            - Include any other text than the estimation
            - Include any other formatting than the estimation format
            """

    return system_prompt.format(examples=examples)


def generate_estimation(
        meeting_summary: str,
) -> Union[ModelResponse, CustomStreamWrapper]:
    """Generate an estimation for a software development project based on a meeting summary."""

    system_prompt = _generate_system_prompt()

    log.info(
        "generating_estimation",
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
    )

    try:
        response = completion(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": meeting_summary},
            ],
            max_tokens=MAX_TOKENS,
            reasoning_effort="none"
        )

        log.info(
            "llm_response_received",
            provider=settings.LLM_PROVIDER,
            model=settings.LLM_MODEL,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

        print(response.model_dump_json(indent=2))

        return response
    except Exception as e:
        log.error("llm_call_failed", error=str(e), provider=settings.LLM_PROVIDER)
        raise LLMServiceError(f"LLM call failed: {e}") from e
