"""Pure grading: a case either lands in the band (or abstains) or it does not."""

from __future__ import annotations

from evals.production.schemas import CaseEvaluation, GoldenCase, Outcome


def evaluate_case(case: GoldenCase, outcome: Outcome, *, arm: str) -> CaseEvaluation:
    """Apply the Session 16 definition of 'good' to one case × one arm."""
    if outcome.skipped:
        return CaseEvaluation(
            case_id=case.id,
            arm=arm,  # type: ignore[arg-type]
            verdict="skipped",
            passed=False,
            expected_engineer_days=case.expected_engineer_days,
            acceptable_range=case.acceptable_range,
            latency_ms=outcome.latency_ms,
            llm_calls=outcome.llm_calls,
            http_status=outcome.http_status,
            error=outcome.skip_reason or outcome.error,
            notes=outcome.skip_reason or "skipped",
        )
    if outcome.throttled:
        return CaseEvaluation(
            case_id=case.id,
            arm=arm,  # type: ignore[arg-type]
            verdict="throttled",
            passed=False,
            expected_engineer_days=case.expected_engineer_days,
            acceptable_range=case.acceptable_range,
            abstained=outcome.abstained,
            abstention_signal=outcome.abstention_signal,
            latency_ms=outcome.latency_ms,
            llm_calls=outcome.llm_calls,
            http_status=outcome.http_status,
            error=outcome.error,
            notes="rate-limited; not counted as a system error",
        )
    if outcome.error:
        return CaseEvaluation(
            case_id=case.id,
            arm=arm,  # type: ignore[arg-type]
            verdict="error",
            passed=False,
            expected_engineer_days=case.expected_engineer_days,
            acceptable_range=case.acceptable_range,
            abstained=outcome.abstained,
            abstention_signal=outcome.abstention_signal,
            latency_ms=outcome.latency_ms,
            llm_calls=outcome.llm_calls,
            http_status=outcome.http_status,
            error=outcome.error,
        )

    source_hit: bool | None = None
    if case.expected_sources_include:
        source_hit = bool(set(case.expected_sources_include) & set(outcome.source_ids))

    if case.expect_abstention:
        passed = bool(outcome.abstained)
        return CaseEvaluation(
            case_id=case.id,
            arm=arm,  # type: ignore[arg-type]
            verdict="passed" if passed else "failed",
            passed=passed,
            expected_engineer_days=case.expected_engineer_days,
            predicted_engineer_days=outcome.engineer_days,
            acceptable_range=case.acceptable_range,
            abs_error=None,
            abstained=outcome.abstained,
            abstention_signal=outcome.abstention_signal,
            source_hit=source_hit,
            latency_ms=outcome.latency_ms,
            llm_calls=outcome.llm_calls,
            http_status=outcome.http_status,
            cost_usd=outcome.cost_usd,
            notes="" if passed else "expected abstention, system estimated",
        )

    # Estimation case: pass ⇔ not abstained AND predicted days inside the band.
    low, high = case.acceptable_range
    predicted = outcome.engineer_days
    in_range = (
        predicted is not None and not outcome.abstained and low <= predicted <= high
    )
    abs_error = None
    if predicted is not None and not outcome.abstained:
        abs_error = abs(predicted - case.expected_engineer_days)
    notes = ""
    if outcome.abstained:
        notes = "abstained on an estimation case"
    elif predicted is None:
        notes = "no engineer_days in the response"
    elif not in_range:
        notes = f"predicted {predicted} outside [{low}, {high}]"
    return CaseEvaluation(
        case_id=case.id,
        arm=arm,  # type: ignore[arg-type]
        verdict="passed" if in_range else "failed",
        passed=bool(in_range),
        expected_engineer_days=case.expected_engineer_days,
        predicted_engineer_days=predicted,
        acceptable_range=case.acceptable_range,
        abs_error=abs_error,
        abstained=outcome.abstained,
        abstention_signal=outcome.abstention_signal,
        source_hit=source_hit,
        latency_ms=outcome.latency_ms,
        llm_calls=outcome.llm_calls,
        http_status=outcome.http_status,
        cost_usd=outcome.cost_usd,
        notes=notes,
    )
