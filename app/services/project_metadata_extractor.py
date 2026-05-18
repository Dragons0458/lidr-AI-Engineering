import instructor
import structlog
from litellm import completion

from app.config import get_settings
from app.errors.llm_error import LLMServiceError
from app.prompts.loader import render_project_metadata_extraction_prompt
from app.schemas.estimation import EstimationRequest
from app.services.sessions import ProjectMetadata

settings = get_settings()
log = structlog.get_logger()

MAX_TOKENS = 700
MAX_RETRIES = 2
_client = instructor.from_litellm(completion)


def extract_project_metadata(
    previous_metadata: ProjectMetadata,
    request: EstimationRequest,
    llm_response: str,
) -> ProjectMetadata:
    """Extract durable project facts from the latest interaction using an LLM."""
    system_prompt, user_prompt = render_project_metadata_extraction_prompt(
        previous_metadata=previous_metadata,
        request=request,
        llm_response=llm_response,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        extracted_metadata = _create_project_metadata_completion(messages)
    except Exception as e:
        log.error("project_metadata_extraction_failed", error=str(e))
        raise LLMServiceError(f"Project metadata extraction failed: {e}") from e

    return _normalize_metadata(extracted_metadata)


# noinspection PyTypeChecker
def _create_project_metadata_completion(
    messages: list[dict[str, str]],
) -> ProjectMetadata:
    return _client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        response_model=ProjectMetadata,
        max_tokens=MAX_TOKENS,
        max_retries=MAX_RETRIES,
        temperature=0,
        reasoning_effort="none",
    )


def _normalize_metadata(extracted_metadata: ProjectMetadata) -> ProjectMetadata:
    return ProjectMetadata(
        project_name=extracted_metadata.project_name,
        assumed_team_size=extracted_metadata.assumed_team_size,
        mentioned_technologies=_normalize_technologies(
            extracted_metadata.mentioned_technologies,
            excluded_technologies=extracted_metadata.excluded_technologies,
        ),
        excluded_technologies=_normalize_technologies(
            extracted_metadata.excluded_technologies
        ),
        agreed_scope=extracted_metadata.agreed_scope,
    )


def _normalize_technologies(
    technologies: list[str],
    excluded_technologies: list[str] | None = None,
) -> list[str]:
    excluded_keys = {
        technology.strip().lower() for technology in excluded_technologies or []
    }
    merged = []
    seen_indexes = {}
    for technology in technologies:
        stripped_technology = technology.strip()
        normalized_key = stripped_technology.lower()
        if not normalized_key or normalized_key in excluded_keys:
            continue
        if normalized_key in seen_indexes:
            merged[seen_indexes[normalized_key]] = stripped_technology
            continue
        seen_indexes[normalized_key] = len(merged)
        merged.append(stripped_technology)
    return merged
