from types import SimpleNamespace

from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
    MetricResult,
)


def test_latency_budget_metric_passes_within_budget() -> None:
    result = LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 3999})

    assert result == MetricResult(
        name="latency_budget",
        score=1.0,
        passed=True,
        details={"latency_ms": 3999, "budget_ms": 4000},
    )


def test_latency_budget_metric_passes_at_boundary() -> None:
    result = LatencyBudgetMetric(budget_ms=4000).evaluate(
        SimpleNamespace(latency_ms=4000)
    )

    assert result.score == 1.0
    assert result.passed is True


def test_latency_budget_metric_fails_over_budget() -> None:
    result = LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 4001})

    assert result.score == 0.0
    assert result.passed is False


def test_cost_budget_metric_passes_at_boundary() -> None:
    result = CostBudgetMetric(budget_usd=0.05).evaluate({"cost_usd": 0.05})

    assert result.score == 1.0
    assert result.passed is True
    assert result.details == {"cost_usd": 0.05, "budget_usd": 0.05}


def test_cost_budget_metric_fails_over_budget() -> None:
    result = CostBudgetMetric(budget_usd=0.05).evaluate({"cost_usd": 0.0501})

    assert result.score == 0.0
    assert result.passed is False


def test_memory_drift_metric_passes_when_fact_is_in_summary() -> None:
    result = MemoryDriftMetric("project name: Nimbus").evaluate(
        {
            "summary": "The latest estimate still tracks PROJECT NAME: NIMBUS.",
            "anchors": [],
            "metadata": {},
        }
    )

    assert result.score == 1.0
    assert result.passed is True
    assert result.details["matched_fields"] == ["summary"]


def test_memory_drift_metric_passes_when_fact_is_in_metadata_alias() -> None:
    result = MemoryDriftMetric("stack includes Flutter").evaluate(
        {
            "summary": "",
            "anchors": [],
            "project_metadata": {
                "project_name": "Atlas Field",
                "mentioned_technologies": ["stack includes Flutter"],
            },
        }
    )

    assert result.score == 1.0
    assert result.passed is True
    assert result.details["matched_fields"] == ["metadata"]


def test_memory_drift_metric_fails_when_fact_is_absent() -> None:
    result = MemoryDriftMetric("budget locked: 80000 EUR").evaluate(
        {
            "summary": "Budget is still unknown.",
            "anchors": ["project name: LedgerFlow"],
            "metadata": {"mentioned_technologies": ["Django"]},
        }
    )

    assert result.score == 0.0
    assert result.passed is False
    assert result.details["matched_fields"] == []
