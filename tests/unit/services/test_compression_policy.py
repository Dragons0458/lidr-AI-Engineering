from unittest.mock import MagicMock

from app.services.compression.anchors import AnchorMatch, AnchorDetector
from app.services.compression.policy import CompressionPolicy
from app.services.compression.summarizer import CumulativeSummarizer
from app.services.sessions import ConversationHistory


def test_overflow_promotes_anchors_and_summarizes_rest() -> None:
    history = ConversationHistory(max_turns=2)
    history.add_message("user", "NDA signed for this project.")
    history.add_message("assistant", "Acknowledged.")
    history.add_message("user", "Add login module.")
    history.add_message("assistant", "Estimated login.")
    history.add_message("user", "Add reporting module.")
    history.add_message("assistant", "Estimated reporting.")
    history.add_message("user", "Add admin panel.")
    history.add_message("assistant", "Estimated admin.")

    detector = MagicMock(spec=AnchorDetector)
    detector.detect.side_effect = [
        AnchorMatch(is_anchor=True, matched_rules=["nda"]),
        AnchorMatch(is_anchor=False, matched_rules=[]),
        AnchorMatch(is_anchor=False, matched_rules=[]),
    ]
    summarizer = MagicMock(spec=CumulativeSummarizer)
    summarizer.summarize.return_value = "Resumen acumulado."

    CompressionPolicy(detector, summarizer).apply(history)

    assert len(history.turns) == 2
    assert history.anchors_count >= 2
    assert history.summary == "Resumen acumulado."
    summarizer.summarize.assert_called_once()


def test_no_op_without_overflow() -> None:
    history = ConversationHistory(max_turns=4)
    history.add_message("user", "One")
    history.add_message("assistant", "Two")

    detector = MagicMock(spec=AnchorDetector)
    summarizer = MagicMock(spec=CumulativeSummarizer)
    CompressionPolicy(detector, summarizer).apply(history)

    detector.detect.assert_not_called()
    summarizer.summarize.assert_not_called()


def test_apply_is_idempotent_when_no_overflow() -> None:
    history = ConversationHistory(max_turns=2)
    history.add_message("user", "A")
    history.add_message("assistant", "B")

    detector = MagicMock(spec=AnchorDetector)
    summarizer = MagicMock(spec=CumulativeSummarizer)
    policy = CompressionPolicy(detector, summarizer)
    policy.apply(history)
    policy.apply(history)
    assert summarizer.summarize.call_count <= 1
