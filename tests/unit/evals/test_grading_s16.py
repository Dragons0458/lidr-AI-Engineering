"""The definition of 'good' for Session 16: in-range or correctly abstained."""

from __future__ import annotations

from evals.production.grading import evaluate_case
from evals.production.schemas import GoldenCase, Outcome


def _case(**overrides) -> GoldenCase:
    payload = {
        "id": "S16-X",
        "title": "t",
        "difficulty": "easy",
        "transcript_path": "x.txt",
        "expected_engineer_days": 40,
        "acceptable_range": [24, 60],
        "expected_sources_include": ["S07-FIN-001"],
        "expect_abstention": False,
    }
    payload.update(overrides)
    return GoldenCase.model_validate(payload)


def test_within_range_passes() -> None:
    ev = evaluate_case(_case(), Outcome(engineer_days=40), arm="rag")
    assert ev.passed is True
    assert ev.abs_error == 0
    assert ev.verdict == "passed"


def test_outside_range_fails() -> None:
    ev = evaluate_case(_case(), Outcome(engineer_days=90), arm="rag")
    assert ev.passed is False
    assert ev.abs_error == 50


def test_abstains_when_it_should() -> None:
    case = _case(
        expect_abstention=True, expected_engineer_days=0, acceptable_range=[0, 0]
    )
    ev = evaluate_case(
        case, Outcome(abstained=True, abstention_signal="explicit"), arm="rag"
    )
    assert ev.passed is True


def test_abstains_when_it_should_not() -> None:
    ev = evaluate_case(
        _case(),
        Outcome(abstained=True, abstention_signal="explicit"),
        arm="rag",
    )
    assert ev.passed is False
    assert "abstained" in ev.notes


def test_responded_when_it_should_abstain() -> None:
    case = _case(
        expect_abstention=True, expected_engineer_days=0, acceptable_range=[0, 0]
    )
    ev = evaluate_case(case, Outcome(engineer_days=20, abstained=False), arm="rag")
    assert ev.passed is False


def test_transport_error() -> None:
    ev = evaluate_case(_case(), Outcome(error="HTTP 502", http_status=502), arm="rag")
    assert ev.passed is False
    assert ev.verdict == "error"


def test_throttled_is_not_a_system_error_verdict() -> None:
    ev = evaluate_case(_case(), Outcome(throttled=True, http_status=429), arm="rag")
    assert ev.verdict == "throttled"
    assert ev.passed is False


def test_source_hit_is_separate_from_passed() -> None:
    ev = evaluate_case(
        _case(),
        Outcome(engineer_days=40, source_ids=["S07-ECO-001"]),
        arm="rag",
    )
    assert ev.passed is True
    assert ev.source_hit is False

    hit = evaluate_case(
        _case(),
        Outcome(engineer_days=40, source_ids=["S07-FIN-001"]),
        arm="rag",
    )
    assert hit.source_hit is True
