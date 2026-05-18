import hashlib
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import escape
import structlog

from app.schemas.estimation import EstimationRequest
from app.services.sessions import ProjectMetadata

_PROMPTS_DIR = Path(__file__).resolve().parent
_ESTIMATION_PROMPTS_ROOT = "estimation"
_PROJECT_METADATA_EXTRACTION_PROMPTS_ROOT = "project_metadata_extraction"
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
) -> tuple[str, str]:
    """Render system and user prompts for estimation requests."""
    template_base_path = f"{_ESTIMATION_PROMPTS_ROOT}/{version}"
    template_context = {
        "request": request,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "description": request.description,
        "attachments": request.attachments or [],
        "project_metadata": format_project_metadata_for_prompt(project_metadata),
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


def render_project_metadata_extraction_prompt(
    previous_metadata: ProjectMetadata,
    request: EstimationRequest,
    llm_response: str,
) -> tuple[str, str]:
    """Render system and user prompts for project metadata extraction."""
    template_context = {
        "previous_metadata_json": previous_metadata.model_dump_json(),
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
