from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)
from app.services.sessions import ConversationHistory, ProjectMetadata


def test_conversation_history_defaults_to_six_turn_sliding_window() -> None:
    history = ConversationHistory()

    for turn_number in range(8):
        history.add_message("user", f"user {turn_number}")
        history.add_message("assistant", f"assistant {turn_number}")

    assert history.max_turns == 6
    assert len(history.turns) == 6
    assert history.turns[0].user.content == "user 2"
    assert history.turns[-1].assistant.content == "assistant 7"


def test_conversation_history_allows_zero_turn_window() -> None:
    history = ConversationHistory(max_turns=0)

    history.add_message("user", "Ignored user request.")
    history.add_message("assistant", "Ignored estimate.")

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


def _build_estimation_request(description: str) -> EstimationRequest:
    return EstimationRequest(
        description=description,
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.LINE_ITEMS,
    )
