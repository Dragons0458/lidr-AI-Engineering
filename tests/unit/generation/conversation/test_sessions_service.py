from app.domain.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)
from app.generation.conversation.store import (
    ChatMessage,
    ConversationHistory,
    ProjectMetadata,
)


def test_conversation_history_trim_restores_sliding_window() -> None:
    history = ConversationHistory()

    for turn_number in range(8):
        history.add_message("user", f"user {turn_number}")
        history.add_message("assistant", f"assistant {turn_number}")

    history.trim_to_max_turns()

    assert history.max_turns == 6
    assert len(history.turns) == 6
    assert history.turns[0].user.content == "user 2"
    assert history.turns[-1].assistant.content == "assistant 7"


def test_conversation_history_allows_zero_turn_window() -> None:
    history = ConversationHistory(max_turns=0)

    history.add_message("user", "Ignored user request.")
    history.add_message("assistant", "Ignored estimate.")
    history.trim_to_max_turns()

    assert list(history.turns) == []


def test_conversation_history_to_messages_list_regenerates_system_prompt() -> None:
    history = ConversationHistory(max_turns=2)
    history.add_message("user", "Previous user request.")
    history.add_message("assistant", "Previous estimate.")
    request = _build_estimation_request("Current request with reporting module.")
    metadata = ProjectMetadata(
        project_name="Portal Clientes",
        mentioned_technologies=["FastAPI"],
    )

    messages = history.to_messages_list(
        request,
        prompt_version="v1",
        project_metadata=metadata,
    )

    assert messages[0]["role"] == "system"
    assert "<project_name>Portal Clientes</project_name>" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Previous user request."}
    assert messages[2] == {"role": "assistant", "content": "Previous estimate."}
    assert messages[-1]["role"] == "user"
    assert "Current request with reporting module." in messages[-1]["content"]


def test_to_messages_list_includes_summary_and_anchors() -> None:
    history = ConversationHistory(max_turns=4)
    history.summary = "Proyecto con login."
    history.anchors.append(ChatMessage(role="user", content="NDA firmado."))
    history.add_message("user", "Última petición con suficiente detalle.")
    history.add_message("assistant", "Estimación reciente.")
    request = _build_estimation_request("Petición actual con módulo de reportes.")

    messages = history.to_messages_list(request, prompt_version="v1")

    assert "[Resumen de la conversación previa]" in messages[1]["content"]
    assert messages[2]["content"] == "NDA firmado."
    assert messages[-1]["role"] == "user"
    assert "Petición actual" in messages[-1]["content"]


def _build_estimation_request(description: str) -> EstimationRequest:
    return EstimationRequest(
        description=description,
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.LINE_ITEMS,
    )
