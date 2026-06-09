import json

from app.domain.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    EstimationResult,
    OutputFormat,
    Phase,
    ProjectType,
)
from app.generation.conversation.transcript import (
    build_acb_turn_context,
    build_conversation_context,
    format_assistant_message_for_context,
)
from app.generation.conversation.store import Session


def _estimation_request(description: str) -> EstimationRequest:
    return EstimationRequest(
        description=description,
        project_type=ProjectType.MOBILE_APP,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )


def test_format_assistant_message_for_context_parses_json_estimation() -> None:
    result = EstimationResult(
        summary="App móvil de recetas de tortas de queso con listado y favoritos.",
        confidence_pct=75,
        phases=[
            Phase(
                name="MVP",
                base_hours=100,
                buffer_hours=10,
                team="2 devs",
                summary="Desarrollo del MVP con listado de recetas y favoritos básicos.",
            )
        ],
        total_base_hours=100,
        total_buffer_hours=10,
        total_hours=110,
        total_cost_eur=6000,
    )
    formatted = format_assistant_message_for_context(result.model_dump_json())
    assert "App móvil de recetas" in formatted
    assert "110h" in formatted


def test_build_conversation_context_includes_prior_turns() -> None:
    session = Session.get_or_create("ctx-test")
    session.history.turns.clear()
    session.history.add_message("user", "Una app para hacer tortas de queso")
    session.history.add_message("assistant", "Estimación previa en markdown breve.")

    context = build_conversation_context(session.history, exclude_latest_user=False)

    assert "tortas de queso" in context
    assert "Estimación previa" in context


def test_build_acb_turn_context_marks_follow_up_and_enriches() -> None:
    session = Session.get_or_create("acb-ctx-test")
    session.history.turns.clear()
    session.history.add_message("user", "Una app para hacer tortas de queso")
    session.history.add_message(
        "assistant",
        json.dumps(
            {
                "summary": "App móvil de recetas de tortas de queso.",
                "confidence_pct": 80,
                "phases": [
                    {
                        "name": "MVP",
                        "base_hours": 80,
                        "buffer_hours": 8,
                        "team": "2 devs",
                        "summary": "MVP con listado de recetas y favoritos para usuarios.",
                    }
                ],
                "total_base_hours": 80,
                "total_buffer_hours": 8,
                "total_hours": 88,
                "total_cost_eur": 5000,
            }
        ),
    )

    enriched, context, is_follow_up = build_acb_turn_context(
        session,
        _estimation_request("¿Se puede hacer con Angular y NestJS?"),
    )

    assert is_follow_up is True
    assert "tortas de queso" in context
    assert "Angular y NestJS" in enriched
    assert "Mensaje actual del usuario" in enriched


def test_build_acb_turn_context_first_turn_is_not_follow_up() -> None:
    session = Session.get_or_create("acb-first-turn")
    session.history.turns.clear()

    enriched, context, is_follow_up = build_acb_turn_context(
        session,
        _estimation_request("Una app para hacer tortas de queso con listado."),
    )

    assert is_follow_up is False
    assert context == ""
    assert enriched == "Una app para hacer tortas de queso con listado."
