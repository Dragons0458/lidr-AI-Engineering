import hashlib
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
import structlog

from app.schemas.estimation import EstimationRequest

_PROMPTS_DIR = Path(__file__).resolve().parent
_ESTIMATION_PROMPTS_ROOT = "estimation"
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
) -> tuple[str, str]:
    """Render system and user prompts for estimation requests."""
    template_base_path = f"{_ESTIMATION_PROMPTS_ROOT}/{version}"
    template_context = {
        "request": request,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "transcript": request.transcript,
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
