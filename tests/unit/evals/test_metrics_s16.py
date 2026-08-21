"""Arm aggregation never divides by zero; p95 is nearest-rank."""

from __future__ import annotations

from evals.production.metrics import aggregate, percentile
from evals.production.schemas import CaseEvaluation, GoldenCase


def _eval(**kwargs) -> CaseEvaluation:
    payload = {
        "case_id": "S16-01",
        "arm": "rag",
        "verdict": "passed",
        "passed": True,
        "expected_engineer_days": 40,
        "predicted_engineer_days": 42,
        "acceptable_range": (24, 60),
        "abs_error": 2.0,
        "latency_ms": 1000.0,
    }
    payload.update(kwargs)
    return CaseEvaluation.model_validate(payload)


def _cases() -> list[GoldenCase]:
    return [
        GoldenCase.model_validate(
            {
                "id": "S16-01",
                "title": "a",
                "difficulty": "easy",
                "transcript_path": "a.txt",
                "expected_engineer_days": 40,
                "acceptable_range": [24, 60],
                "expect_abstention": False,
            }
        ),
        GoldenCase.model_validate(
            {
                "id": "S16-06",
                "title": "b",
                "difficulty": "abstention",
                "transcript_path": "b.txt",
                "expected_engineer_days": 0,
                "acceptable_range": [0, 0],
                "expect_abstention": True,
            }
        ),
    ]


def test_percentile_nearest_rank() -> None:
    assert percentile([], 95) is None
    assert percentile([10, 20, 30, 40, 50, 60, 70], 95) == 70.0


def test_within_range_mae_p95_error_rate() -> None:
    evaluations = [
        _eval(
            case_id="S16-01",
            passed=True,
            verdict="passed",
            abs_error=2.0,
            latency_ms=100,
        ),
        _eval(
            case_id="S16-01b",
            passed=False,
            verdict="failed",
            predicted_engineer_days=90,
            abs_error=50.0,
            latency_ms=200,
        ),
        _eval(
            case_id="S16-06",
            passed=True,
            verdict="passed",
            expected_engineer_days=0,
            predicted_engineer_days=None,
            abs_error=None,
            latency_ms=50,
        ),
    ]
    # S16-01b is not in the golden set; treat extra estimation-like rows via a third case.
    cases = _cases() + [
        GoldenCase.model_validate(
            {
                "id": "S16-01b",
                "title": "c",
                "difficulty": "easy",
                "transcript_path": "c.txt",
                "expected_engineer_days": 40,
                "acceptable_range": [24, 60],
                "expect_abstention": False,
            }
        )
    ]
    report = aggregate(evaluations, cases, arm="rag")
    assert report.within_range_rate == 0.5
    assert report.mean_absolute_error == 26.0
    assert report.abstention_correct is True
    assert report.p95_latency_ms == 200.0
    assert report.p95_n == 3
    assert report.error_rate == 0.0


def test_zero_estimation_cases() -> None:
    cases = [_cases()[1]]
    evaluations = [
        _eval(
            case_id="S16-06",
            passed=True,
            expected_engineer_days=0,
            predicted_engineer_days=None,
            abs_error=None,
        )
    ]
    report = aggregate(evaluations, cases, arm="rag")
    assert report.within_range_rate is None
    assert report.mean_absolute_error is None
    assert report.abstention_correct is True


def test_all_abstained_estimation_cases() -> None:
    cases = [_cases()[0]]
    evaluations = [
        _eval(
            case_id="S16-01",
            passed=False,
            verdict="failed",
            predicted_engineer_days=None,
            abs_error=None,
            abstained=True,
        )
    ]
    report = aggregate(evaluations, cases, arm="rag")
    assert report.within_range_rate == 0.0
    assert report.mean_absolute_error is None


def test_all_errors() -> None:
    cases = [_cases()[0]]
    evaluations = [
        _eval(
            case_id="S16-01",
            passed=False,
            verdict="error",
            error="boom",
            abs_error=None,
        )
    ]
    report = aggregate(evaluations, cases, arm="rag")
    assert report.error_rate == 1.0
    assert report.within_range_rate is None


def test_empty_evaluations() -> None:
    report = aggregate([], [], arm="graph")
    assert report.n_cases == 0
    assert report.within_range_rate is None
    assert report.error_rate is None
    assert report.p95_latency_ms is None
