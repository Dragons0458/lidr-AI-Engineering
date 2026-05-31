from unittest.mock import MagicMock

from app.services.compression.summarizer import CumulativeSummarizer, _SummaryEnvelope
from app.services.sessions import ChatMessage


def test_summarize_folds_evicted_messages() -> None:
    wrapper = MagicMock()
    wrapper.complete_structured_chat.return_value = (
        _SummaryEnvelope(summary="Resumen nuevo."),
        {},
    )
    summarizer = CumulativeSummarizer(wrapper, "gpt-4o-mini")
    result = summarizer.summarize(
        previous_summary="Resumen previo.",
        evicted=[ChatMessage(role="user", content="Turno antiguo.")],
    )
    assert result == "Resumen nuevo."


def test_summarize_error_keeps_previous() -> None:
    wrapper = MagicMock()
    wrapper.complete_structured_chat.side_effect = RuntimeError("llm down")
    summarizer = CumulativeSummarizer(wrapper, "gpt-4o-mini")
    result = summarizer.summarize(
        previous_summary="Resumen previo.",
        evicted=[ChatMessage(role="user", content="Turno antiguo.")],
    )
    assert result == "Resumen previo."
