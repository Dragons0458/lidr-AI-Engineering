from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class MetricResult:
    name: str
    score: float
    passed: bool
    details: dict[str, Any]


class LatencyBudgetMetric:
    """1.0 if latency_ms is within budget; 0.0 otherwise."""

    name = "latency_budget"

    def __init__(self, budget_ms: int):
        self.budget_ms = budget_ms

    def evaluate(self, observation: Any) -> MetricResult:
        latency_ms = int(_get_value(observation, "latency_ms", 0) or 0)
        passed = latency_ms <= self.budget_ms
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={
                "latency_ms": latency_ms,
                "budget_ms": self.budget_ms,
            },
        )


class CostBudgetMetric:
    """1.0 if cost_usd is within budget; 0.0 otherwise."""

    name = "cost_budget"

    def __init__(self, budget_usd: float):
        self.budget_usd = budget_usd

    def evaluate(self, observation: Any) -> MetricResult:
        cost_usd = float(_get_value(observation, "cost_usd", 0.0) or 0.0)
        passed = cost_usd <= self.budget_usd
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={
                "cost_usd": cost_usd,
                "budget_usd": self.budget_usd,
            },
        )


class MemoryDriftMetric:
    """1.0 if the declared fact appears in the selected session fields."""

    name = "memory_drift"

    def __init__(
        self,
        fact: str,
        where: Iterable[str] = ("summary", "anchors", "metadata"),
    ):
        self.fact = fact
        self.where = tuple(where)

    def evaluate(self, session_snapshot: Any) -> MetricResult:
        fact = self.fact.casefold()
        matched_fields = [
            field
            for field in self.where
            if fact in _field_text(session_snapshot, field).casefold()
        ]
        passed = bool(matched_fields)

        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={
                "fact": self.fact,
                "where": list(self.where),
                "matched_fields": matched_fields,
            },
        )


def _get_value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)

    return getattr(source, key, default)


def _field_text(snapshot: Any, field: str) -> str:
    if field == "metadata":
        value = _get_value(snapshot, "metadata")
        if value is None:
            value = _get_value(snapshot, "project_metadata", {})
        return _stringify(value)

    return _stringify(_get_value(snapshot, field, ""))


def _stringify(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, Mapping):
        return " ".join(
            f"{_stringify(key)} {_stringify(item)}" for key, item in value.items()
        )

    if isinstance(value, Iterable):
        return " ".join(_stringify(item) for item in value)

    return str(value)
