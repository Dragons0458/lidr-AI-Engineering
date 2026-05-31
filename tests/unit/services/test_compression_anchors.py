from unittest.mock import MagicMock

from app.services.compression.anchors import AnchorDetector
from app.services.sessions import ChatMessage


def test_heuristic_matches_nda() -> None:
    detector = AnchorDetector("heuristic")
    match = detector.detect(ChatMessage(role="user", content="Please respect our NDA."))
    assert match.is_anchor is True
    assert "nda" in match.matched_rules


def test_heuristic_matches_compliance() -> None:
    detector = AnchorDetector("heuristic")
    match = detector.detect(
        ChatMessage(role="user", content="HIPAA compliance is mandatory.")
    )
    assert match.is_anchor is True


def test_heuristic_non_anchor() -> None:
    detector = AnchorDetector("heuristic")
    match = detector.detect(
        ChatMessage(role="user", content="Add a reporting dashboard.")
    )
    assert match.is_anchor is False


def test_llm_mode_falls_back_on_failure() -> None:
    wrapper = MagicMock()
    detector = AnchorDetector(
        "llm",
        llm_wrapper=wrapper,
        llm_model=None,
    )
    match = detector.detect(ChatMessage(role="user", content="NDA signed today."))
    assert match.is_anchor is True
