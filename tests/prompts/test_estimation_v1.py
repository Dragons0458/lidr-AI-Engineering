from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)


def _build_request(
    *,
    transcript: str = "Project with auth, billing, and reporting requirements.",
    output_format: OutputFormat = OutputFormat.LINE_ITEMS,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
) -> EstimationRequest:
    return EstimationRequest(
        transcript=transcript,
        project_type=ProjectType.WEB_SAAS,
        detail_level=detail_level,
        output_format=output_format,
    )


def test_user_prompt_includes_literal_project_description_inside_project_description_tag() -> None:
    transcript = "Need <module-a> & <module-b> with SSO and CSV exports."
    request = _build_request(transcript=transcript)

    _, user_prompt = render_estimation_prompt(request, version="v1")

    assert "<project_description><![CDATA[" in user_prompt
    assert transcript in user_prompt
    assert "]]></project_description>" in user_prompt


def test_system_prompt_changes_format_instructions_for_phases_table_vs_narrative() -> None:
    phases_table_request = _build_request(output_format=OutputFormat.PHASES_TABLE)
    narrative_request = _build_request(output_format=OutputFormat.NARRATIVE)

    phases_table_system, _ = render_estimation_prompt(phases_table_request, version="v1")
    narrative_system, _ = render_estimation_prompt(narrative_request, version="v1")

    assert "<selected_output_format>phases_table</selected_output_format>" in phases_table_system
    assert "Required columns: Phase, Tasks, Hours, Team." in phases_table_system

    assert "<selected_output_format>narrative</selected_output_format>" in narrative_system
    assert "Required columns: Phase, Tasks, Hours, Team." not in narrative_system


def test_system_prompt_includes_detailed_assumptions_instruction_but_not_for_summary() -> None:
    detailed_request = _build_request(detail_level=DetailLevel.DETAILED)
    summary_request = _build_request(detail_level=DetailLevel.SUMMARY)

    detailed_system, _ = render_estimation_prompt(detailed_request, version="v1")
    summary_system, _ = render_estimation_prompt(summary_request, version="v1")

    assert "<selected_detail_level>detailed</selected_detail_level>" in detailed_system
    assert "Include explicit assumptions, risks, and dependencies." in detailed_system

    assert "<selected_detail_level>summary</selected_detail_level>" in summary_system
    assert "Include explicit assumptions, risks, and dependencies." not in summary_system


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
