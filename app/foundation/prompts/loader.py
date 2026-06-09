import hashlib
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import escape
import structlog

from app.foundation.prompts.examples import build_prompt_examples
from app.domain.schemas.critic import CriticFeedback
from app.domain.schemas.estimation import (
    EstimationResult,
    ExampleFormat,
    EstimationRequest,
)
from app.generation.conversation.store import ChatMessage, ProjectMetadata

_PROMPTS_DIR = Path(__file__).resolve().parent
_ESTIMATION_PROMPTS_ROOT = "estimation"
_PROJECT_METADATA_EXTRACTION_PROMPTS_ROOT = "project_metadata_extraction"
_REQUIREMENTS_EXTRACTION_PROMPTS_ROOT = "requirements_extraction"
log = structlog.get_logger()

_jinja_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = "v1",
    project_metadata: ProjectMetadata | None = None,
    use_examples: bool | None = None,
    num_examples: int | None = None,
    example_format: ExampleFormat | None = None,
) -> tuple[str, str]:
    """Render system and user prompts for estimation requests."""
    template_base_path = f"{_ESTIMATION_PROMPTS_ROOT}/{version}"
    selected_use_examples = (
        request.use_examples if use_examples is None else use_examples
    )
    selected_num_examples = (
        request.num_examples if num_examples is None else num_examples
    )
    selected_example_format = (
        request.example_format if example_format is None else example_format
    )
    template_context = {
        "request": request,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "description": request.description,
        "inline_cleaning": request.preprocessing == "inline_cleaning",
        "attachments": request.attachments or [],
        "project_metadata": format_project_metadata_for_prompt(project_metadata),
        "examples": build_prompt_examples(
            use_examples=selected_use_examples,
            num_examples=selected_num_examples,
            example_format=selected_example_format,
        ),
    }

    system_prompt = _jinja_env.get_template(f"{template_base_path}/system.j2").render(
        **template_context
    )
    user_prompt = _jinja_env.get_template(f"{template_base_path}/user.j2").render(
        **template_context
    )

    rendered_content = f"{system_prompt}\n\n{user_prompt}"
    rendered_hash = hashlib.sha256(rendered_content.encode("utf-8")).hexdigest()
    log.info(
        "estimation_prompt_rendered",
        prompt_version=version,
        prompt_hash=rendered_hash,
        rendered_content=rendered_content,
    )

    return system_prompt, user_prompt


def render_requirements_extraction_prompt(
    request: EstimationRequest,
) -> tuple[str, str]:
    """Render system and user prompts for two-phase requirements extraction."""
    template_context = {
        "description": request.description,
        "attachments": request.attachments or [],
        "project_type": request.project_type.value,
    }
    system_prompt = _jinja_env.get_template(
        f"{_REQUIREMENTS_EXTRACTION_PROMPTS_ROOT}/system.j2"
    ).render(**template_context)
    user_prompt = _jinja_env.get_template(
        f"{_REQUIREMENTS_EXTRACTION_PROMPTS_ROOT}/user.j2"
    ).render(**template_context)

    return system_prompt, user_prompt


def render_project_metadata_extraction_prompt(
    previous_metadata: ProjectMetadata,
    request: EstimationRequest,
    llm_response: str,
) -> tuple[str, str]:
    """Render system and user prompts for project metadata extraction."""
    template_context = {
        "previous_metadata_json": previous_metadata.model_dump_json(
            exclude_none=True,
            exclude_defaults=True,
        ),
        "description": request.description,
        "attachments": request.attachments or [],
        "llm_response": llm_response,
    }
    system_prompt = _jinja_env.get_template(
        f"{_PROJECT_METADATA_EXTRACTION_PROMPTS_ROOT}/system.j2"
    ).render(**template_context)
    user_prompt = _jinja_env.get_template(
        f"{_PROJECT_METADATA_EXTRACTION_PROMPTS_ROOT}/user.j2"
    ).render(**template_context)

    return system_prompt, user_prompt


def format_project_metadata_for_prompt(
    project_metadata: ProjectMetadata | None,
) -> str:
    """Format known project facts as prompt-safe XML-like content."""
    if project_metadata is None or _is_project_metadata_empty(project_metadata):
        return ""

    lines = []
    if project_metadata.project_name:
        lines.append(
            f"<project_name>{escape(project_metadata.project_name)}</project_name>"
        )
    if project_metadata.assumed_team_size:
        lines.append(
            f"<assumed_team_size>{project_metadata.assumed_team_size}</assumed_team_size>"
        )
    if project_metadata.mentioned_technologies:
        lines.append("<mentioned_technologies>")
        for technology in project_metadata.mentioned_technologies:
            lines.append(f"<technology>{escape(technology)}</technology>")
        lines.append("</mentioned_technologies>")
    if project_metadata.excluded_technologies:
        lines.append("<excluded_technologies>")
        for technology in project_metadata.excluded_technologies:
            lines.append(f"<technology>{escape(technology)}</technology>")
        lines.append("</excluded_technologies>")
    if project_metadata.agreed_scope:
        lines.append(
            f"<agreed_scope>{escape(project_metadata.agreed_scope)}</agreed_scope>"
        )

    return "\n".join(lines)


def render_structured_estimation_prompt(
    *,
    request: EstimationRequest,
    metadata: ProjectMetadata | None = None,
    tier: str = "default",
    critic_feedback: CriticFeedback | None = None,
    conversation_context: str | None = None,
    is_follow_up: bool = False,
    enriched_description: str | None = None,
    version: str = "v3",
    use_examples: bool | None = None,
    num_examples: int | None = None,
    example_format: ExampleFormat | None = None,
) -> tuple[str, str]:
    """Render system and user prompts for structured (ACB) estimation."""
    template_base_path = f"{_ESTIMATION_PROMPTS_ROOT}/{version}"
    selected_use_examples = (
        request.use_examples if use_examples is None else use_examples
    )
    selected_num_examples = (
        request.num_examples if num_examples is None else num_examples
    )
    selected_example_format = (
        request.example_format if example_format is None else example_format
    )
    description = enriched_description or request.description
    template_context = {
        "request": request,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "description": description,
        "latest_message": request.description,
        "conversation_context": conversation_context or "",
        "is_follow_up": is_follow_up,
        "attachments": request.attachments or [],
        "project_metadata": format_project_metadata_for_prompt(metadata),
        "tier": tier,
        "critic_feedback": critic_feedback,
        "examples": build_prompt_examples(
            use_examples=selected_use_examples,
            num_examples=selected_num_examples,
            example_format=selected_example_format,
        ),
    }
    system_prompt = _jinja_env.get_template(f"{template_base_path}/system.j2").render(
        **template_context
    )
    user_prompt = _jinja_env.get_template(f"{template_base_path}/user.j2").render(
        **template_context
    )
    return system_prompt, user_prompt


def render_critic_prompt(
    *,
    transcript: str,
    metadata: ProjectMetadata,
    tier: str,
    result: EstimationResult,
    version: str = "v1",
) -> tuple[str, str]:
    template_base = f"critic/{version}"
    context = {
        "transcript": transcript,
        "project_metadata": format_project_metadata_for_prompt(metadata),
        "tier": tier,
        "result": result,
    }
    system_prompt = _jinja_env.get_template(f"{template_base}/system.j2").render(
        **context
    )
    user_prompt = _jinja_env.get_template(f"{template_base}/user.j2").render(**context)
    return system_prompt, user_prompt


def render_conversation_summary_prompt(
    *,
    previous_summary: str,
    evicted: list[ChatMessage],
    version: str = "v1",
) -> tuple[str, str]:
    template_base = f"conversation_summary/{version}"
    context = {
        "previous_summary": previous_summary,
        "evicted": evicted,
    }
    system_prompt = _jinja_env.get_template(f"{template_base}/system.j2").render(
        **context
    )
    user_prompt = _jinja_env.get_template(f"{template_base}/user.j2").render(**context)
    return system_prompt, user_prompt


def _is_project_metadata_empty(project_metadata: ProjectMetadata) -> bool:
    return not any(
        [
            project_metadata.project_name,
            project_metadata.assumed_team_size,
            project_metadata.mentioned_technologies,
            project_metadata.excluded_technologies,
            project_metadata.agreed_scope,
        ]
    )
