from app.foundation.prompts.loader import (
    _jinja_env,
    render_agent_hours_recovery_prompt,
    render_agent_legacy_prompts,
    render_agent_structure_prompt,
    render_estimation_prompt,
    render_project_metadata_extraction_prompt,
    render_structured_estimation_prompt,
)
from jinja2 import UndefinedError
import pytest
from structlog.testing import capture_logs

from app.generation.agentic.agent_schemas import AgentTaskRef
from app.domain.schemas.attachments import AttachmentText
from app.domain.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)
from app.generation.conversation.store import ProjectMetadata


def _build_request(
    *,
    description: str = "Project with auth, billing, and reporting requirements.",
    output_format: OutputFormat = OutputFormat.LINE_ITEMS,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
) -> EstimationRequest:
    return EstimationRequest(
        description=description,
        project_type=ProjectType.WEB_SAAS,
        detail_level=detail_level,
        output_format=output_format,
    )


def test_inline_cleaning_flag_renders_transcription_cleaning_block() -> None:
    request = _build_request().model_copy(update={"preprocessing": "inline_cleaning"})

    system_prompt, _ = render_estimation_prompt(request, version="v1")

    assert "<transcription_cleaning>" in system_prompt
    assert "Extract ONLY the functional and technical requirements" in system_prompt


def test_user_prompt_includes_literal_project_description_inside_project_description_tag() -> (
    None
):
    description = "Need <module-a> & <module-b> with SSO and CSV exports."
    request = _build_request(description=description)

    _, user_prompt = render_estimation_prompt(request, version="v1")

    assert "<project_description><![CDATA[" in user_prompt
    assert description in user_prompt
    assert "]]></project_description>" in user_prompt


def test_system_prompt_changes_format_instructions_for_phases_table_vs_narrative() -> (
    None
):
    phases_table_request = _build_request(output_format=OutputFormat.PHASES_TABLE)
    narrative_request = _build_request(output_format=OutputFormat.NARRATIVE)

    phases_table_system, _ = render_estimation_prompt(
        phases_table_request, version="v1"
    )
    narrative_system, _ = render_estimation_prompt(narrative_request, version="v1")

    assert (
        "<selected_output_format>phases_table</selected_output_format>"
        in phases_table_system
    )
    assert "Required columns: Phase, Tasks, Hours, Team." in phases_table_system

    assert (
        "<selected_output_format>narrative</selected_output_format>" in narrative_system
    )
    assert "Required columns: Phase, Tasks, Hours, Team." not in narrative_system


def test_system_prompt_includes_detailed_assumptions_instruction_but_not_for_summary() -> (
    None
):
    detailed_request = _build_request(detail_level=DetailLevel.DETAILED)
    summary_request = _build_request(detail_level=DetailLevel.SUMMARY)

    detailed_system, _ = render_estimation_prompt(detailed_request, version="v1")
    summary_system, _ = render_estimation_prompt(summary_request, version="v1")

    assert "<selected_detail_level>detailed</selected_detail_level>" in detailed_system
    assert "Include explicit assumptions, risks, and dependencies." in detailed_system

    assert "<selected_detail_level>summary</selected_detail_level>" in summary_system
    assert (
        "Include explicit assumptions, risks, and dependencies." not in summary_system
    )


def test_render_estimation_prompt_v2_contains_risk_aware_sections() -> None:
    request = _build_request(
        output_format=OutputFormat.PHASES_TABLE,
        detail_level=DetailLevel.DETAILED,
    )

    system_prompt, user_prompt = render_estimation_prompt(request, version="v2")

    assert "risk-aware planning" in system_prompt
    assert "<planning_principles>" in system_prompt
    assert "Buffer Hours" in system_prompt
    assert "<project_description><![CDATA[" in user_prompt


def test_system_prompt_rejects_out_of_scope_weak_input() -> None:
    request = _build_request(description="Estimar el proyecto.")

    v1_system_prompt, _ = render_estimation_prompt(request, version="v1")
    v2_system_prompt, _ = render_estimation_prompt(request, version="v2")

    for system_prompt in (v1_system_prompt, v2_system_prompt):
        assert "<scope>" in system_prompt
        assert "Out of scope:" in system_prompt
        assert "preliminary discovery/MVP estimate" not in system_prompt


def test_system_prompt_renders_reference_projects_when_present() -> None:
    request = _build_request().model_copy(
        update={
            "reference_projects": [
                {
                    "name": "Billing MVP",
                    "summary": "Project focused on subscriptions, checkout and invoices.",
                    "estimated_hours": 280,
                    "team": "2 backend, 1 frontend",
                    "outcome": "Released in 8 weeks",
                }
            ]
        }
    )

    v1_system_prompt, _ = render_estimation_prompt(request, version="v1")
    v2_system_prompt, _ = render_estimation_prompt(request, version="v2")

    assert "<reference_projects>" in v1_system_prompt
    assert "<name>Billing MVP</name>" in v1_system_prompt
    assert "<estimated_hours>280</estimated_hours>" in v1_system_prompt

    assert "<reference_projects>" in v2_system_prompt
    assert "<name>Billing MVP</name>" in v2_system_prompt


def test_system_prompt_renders_empty_project_metadata_block_by_default() -> None:
    request = _build_request()

    system_prompt, _ = render_estimation_prompt(request, version="v1")

    metadata_block = system_prompt.rsplit("<project_metadata>", maxsplit=1)[1].split(
        "</project_metadata>"
    )[0]
    assert metadata_block.strip() == ""


def test_system_prompt_renders_known_project_metadata_when_present() -> None:
    request = _build_request()
    project_metadata = ProjectMetadata(
        project_name="Portal Clientes",
        assumed_team_size=3,
        mentioned_technologies=["FastAPI", "React"],
        excluded_technologies=["Firebase"],
        agreed_scope="Autenticacion, facturacion y reportes.",
    )

    system_prompt, _ = render_estimation_prompt(
        request, version="v2", project_metadata=project_metadata
    )

    assert "<project_metadata>" in system_prompt
    assert "<project_name>Portal Clientes</project_name>" in system_prompt
    assert "<assumed_team_size>3</assumed_team_size>" in system_prompt
    assert "<technology>FastAPI</technology>" in system_prompt
    assert "<technology>React</technology>" in system_prompt
    assert "<excluded_technologies>" in system_prompt
    assert "<technology>Firebase</technology>" in system_prompt
    assert (
        "<agreed_scope>Autenticacion, facturacion y reportes.</agreed_scope>"
        in system_prompt
    )


def test_project_metadata_extraction_prompt_renders_from_templates() -> None:
    request = _build_request(
        description="Portal Clientes con React, FastAPI y equipo de 3 personas."
    )
    previous_metadata = ProjectMetadata(mentioned_technologies=["PostgreSQL"])

    system_prompt, user_prompt = render_project_metadata_extraction_prompt(
        previous_metadata=previous_metadata,
        request=request,
        llm_response="Estimacion para autenticacion y reportes.",
    )

    assert "You extract durable project facts" in system_prompt
    assert "<previous_project_metadata>" in user_prompt
    assert '"mentioned_technologies":["PostgreSQL"]' in user_prompt
    assert "<latest_project_description><![CDATA[" in user_prompt
    assert "Portal Clientes con React, FastAPI" in user_prompt
    assert "<latest_llm_response><![CDATA[" in user_prompt


def test_project_metadata_extraction_prompt_omits_empty_metadata_fields() -> None:
    request = _build_request(description="Nueva iteracion del proyecto.")
    previous_metadata = ProjectMetadata(project_name="Portal Clientes")

    _, user_prompt = render_project_metadata_extraction_prompt(
        previous_metadata=previous_metadata,
        request=request,
        llm_response="Mantener alcance actual.",
    )

    assert "<previous_project_metadata>" in user_prompt
    assert '"project_name":"Portal Clientes"' in user_prompt
    assert '"assumed_team_size"' not in user_prompt
    assert '"mentioned_technologies"' not in user_prompt
    assert '"excluded_technologies"' not in user_prompt
    assert '"agreed_scope"' not in user_prompt


def test_user_prompt_renders_attachments_with_clear_separator() -> None:
    request = _build_request().model_copy(
        update={
            "attachments": [
                AttachmentText(
                    filename="scope.txt",
                    content="Authentication and reporting requirements.",
                )
            ]
        }
    )

    _, user_prompt = render_estimation_prompt(request, version="v1")

    assert "<attachments>" in user_prompt
    assert '<attachment filename="scope.txt"><![CDATA[' in user_prompt
    assert "--- attachment: scope.txt ---" in user_prompt
    assert "Authentication and reporting requirements." in user_prompt


def test_structured_v3_follow_up_includes_conversation_context() -> None:
    request = _build_request(description="¿Se puede hacer con Angular y NestJS?")

    system_prompt, user_prompt = render_structured_estimation_prompt(
        request=request,
        conversation_context="Usuario: app de tortas de queso\nAsistente: MVP 110h",
        is_follow_up=True,
        enriched_description=(
            "Contexto previo...\n\nMensaje actual: ¿Se puede hacer con Angular y NestJS?"
        ),
    )

    assert "multi-turn" in system_prompt.lower() or "follow-up" in system_prompt.lower()
    assert "<conversation_context>" in user_prompt
    assert "<latest_user_message>" in user_prompt
    assert "Angular y NestJS" in user_prompt


def test_agent_structure_prompt_separates_brief_and_optional_persona():
    brief = {"function": "CONFIDENTIAL_BRIEF"}
    system, user = render_agent_structure_prompt(brief, "PRIVATE_PERSONA")
    assert "PRIVATE_PERSONA" in system
    assert "PRIVATE_PERSONA" not in user
    assert "CONFIDENTIAL_BRIEF" in user
    assert "CONFIDENTIAL_BRIEF" not in system
    system_without_persona, _ = render_agent_structure_prompt(brief)
    assert "Additional working persona" not in system_without_persona


def test_hours_recovery_prompt_lists_flags_and_prohibits_free_hours():
    task = AgentTaskRef(
        task_ref="task-0",
        module="Core",
        task="Build",
        description="Implementation",
        reason="no historical match",
    )
    system, user = render_agent_hours_recovery_prompt([task], "Careful reviewer")
    assert "derive_task_hours" in system
    assert "Never\ninvent or return free-form hours" in system
    assert "Careful reviewer" in system
    assert "task-0" in user
    assert "no historical match" in user


def test_legacy_prompts_render_transcript_and_final_instruction():
    system, initial, final = render_agent_legacy_prompts("SECRET_TRANSCRIPT")
    assert "search_budgets" in system
    assert initial.strip() == "SECRET_TRANSCRIPT"
    assert "total_hours" in final


def test_prompt_environment_uses_strict_undefined():
    with pytest.raises(UndefinedError):
        _jinja_env.from_string("{{ missing }}").render()


def test_agent_prompt_logs_never_include_sensitive_content():
    transcript = "SENSITIVE_TRANSCRIPT_123"
    persona = "SENSITIVE_PERSONA_456"
    task = AgentTaskRef(
        task_ref="task-secret",
        module="Secret",
        task="Hidden",
        reason="private reason",
    )
    with capture_logs() as logs:
        render_agent_structure_prompt({"function": transcript}, persona)
        render_agent_hours_recovery_prompt([task], persona)
        render_agent_legacy_prompts(transcript)
    serialized = repr(logs)
    assert transcript not in serialized
    assert persona not in serialized
    assert "rendered_content" not in serialized
    assert all("prompt_hash" in event for event in logs)
