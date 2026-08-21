"""Aggregate per-case evaluations into an arm report. Degenerate inputs stay None."""

from __future__ import annotations

import math

from evals.production.schemas import ArmName, ArmReport, CaseEvaluation, GoldenCase


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile. ``p`` is in 0..100. Empty → None."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil((p / 100.0) * len(ordered)))
    return float(ordered[rank - 1])


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def aggregate(
    evaluations: list[CaseEvaluation],
    cases: list[GoldenCase],
    *,
    arm: ArmName,
    skipped: bool = False,
    skip_reason: str | None = None,
) -> ArmReport:
    """Build the Session 16 arm metrics. Never raises ``ZeroDivisionError``."""
    n = len(evaluations)
    if skipped:
        return ArmReport(
            arm=arm,
            skipped=True,
            skip_reason=skip_reason,
            n_cases=n,
            evaluations=evaluations,
        )

    estimation_ids = {case.id for case in cases if not case.expect_abstention}
    abstention_ids = {case.id for case in cases if case.expect_abstention}
    estimation = [ev for ev in evaluations if ev.case_id in estimation_ids]
    abstention = [ev for ev in evaluations if ev.case_id in abstention_ids]

    n_passed = sum(1 for ev in evaluations if ev.passed)
    n_failed = sum(1 for ev in evaluations if ev.verdict == "failed")
    n_throttled = sum(1 for ev in evaluations if ev.verdict == "throttled")
    n_error = sum(1 for ev in evaluations if ev.verdict == "error")

    estimation_gradable = [
        ev for ev in estimation if ev.verdict in {"passed", "failed"}
    ]
    within = (
        (sum(1 for ev in estimation_gradable if ev.passed) / len(estimation_gradable))
        if estimation_gradable
        else None
    )
    abs_errors = [
        ev.abs_error for ev in estimation_gradable if ev.abs_error is not None
    ]
    abstention_gradable = [
        ev for ev in abstention if ev.verdict in {"passed", "failed"}
    ]
    abstention_correct = (
        all(ev.passed for ev in abstention_gradable) if abstention_gradable else None
    )

    latencies = [ev.latency_ms for ev in evaluations if ev.verdict != "skipped"]
    counted_for_error = [
        ev for ev in evaluations if ev.verdict not in {"throttled", "skipped"}
    ]
    error_rate = (n_error / len(counted_for_error)) if counted_for_error else None
    abstained_frac = (sum(1 for ev in evaluations if ev.abstained) / n) if n else None
    source_cases = [
        ev
        for ev in evaluations
        if ev.source_hit is not None and ev.verdict in {"passed", "failed"}
    ]
    source_recall = (
        (sum(1 for ev in source_cases if ev.source_hit) / len(source_cases))
        if source_cases
        else None
    )
    costs = [ev.cost_usd for ev in evaluations if ev.cost_usd is not None]
    return ArmReport(
        arm=arm,
        n_cases=n,
        n_estimation=len(estimation),
        n_abstention=len(abstention),
        n_passed=n_passed,
        n_failed=n_failed,
        n_throttled=n_throttled,
        n_error=n_error,
        within_range_rate=within,
        mean_absolute_error=_mean(abs_errors),
        abstention_correct=abstention_correct,
        mean_latency_ms=_mean(latencies),
        p95_latency_ms=percentile(latencies, 95),
        p95_n=len(latencies),
        error_rate=error_rate,
        abstention_rate=abstained_frac,
        source_recall=source_recall,
        total_cost_usd=sum(costs) if costs else None,
        mean_cost_usd=_mean(costs),
        evaluations=evaluations,
    )


def ab_compare(rag: ArmReport | None, graph: ArmReport | None) -> dict:
    """Numeric A/B block. The product verdict is written by a human, not here."""

    def _delta(left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return left - right

    def _ratio(left: float | None, right: float | None) -> float | None:
        if left is None or right is None or right == 0:
            return None
        return left / right

    rag_ids = {ev.case_id: ev.abstained for ev in (rag.evaluations if rag else [])}
    graph_ids = {
        ev.case_id: ev.abstained for ev in (graph.evaluations if graph else [])
    }
    gap = sorted(
        case_id
        for case_id in set(rag_ids) & set(graph_ids)
        if rag_ids[case_id] != graph_ids[case_id]
    )
    return {
        "quality_delta": _delta(
            rag.within_range_rate if rag else None,
            graph.within_range_rate if graph else None,
        ),
        "cost_ratio": _ratio(
            rag.total_cost_usd if rag else None,
            graph.total_cost_usd if graph else None,
        ),
        "latency_ratio": _ratio(
            rag.mean_latency_ms if rag else None,
            graph.mean_latency_ms if graph else None,
        ),
        "abstention_gap": gap,
        "verdict": None,
    }
