import csv
from datetime import datetime
from types import SimpleNamespace

import pytest

from app.domain.schemas.estimation import EstimationResponse, TokenUsage
from app.generation.conversation.store import Session
from evals.stress.run import _evaluate_memory, run_stress


def test_evaluate_memory_scores_previous_facts() -> None:
    score, passed = _evaluate_memory(
        facts=("project name: Nimbus", "scope includes authentication"),
        snapshot={
            "summary": "PROJECT NAME: NIMBUS",
            "anchors": [],
            "metadata": {"agreed_scope": "missing"},
        },
    )

    assert score == 0.5
    assert passed is False


@pytest.mark.anyio
async def test_run_stress_writes_csv_without_llm(monkeypatch, tmp_path) -> None:
    Session.clear_all()

    def fake_generate_estimation(
        request, prompt_version="v1", project_metadata=None, messages=None, **kwargs
    ):
        return SimpleNamespace()

    def fake_format_response(response, prompt_version="v1"):
        return EstimationResponse(
            estimation="Estimated work breakdown for project name: Nimbus",
            model="test-model",
            provider="test-provider",
            timestamp=datetime(2026, 5, 18),
            usage=TokenUsage(
                tokens_used=30,
                input_tokens=20,
                output_tokens=10,
                cost_estimate=0.02,
            ),
            prompt_version=prompt_version,
            latency_ms=123,
        )

    def fake_extract_project_metadata(previous_metadata, request, llm_response):
        return previous_metadata

    monkeypatch.setattr(
        "app.api.sessions.generate_estimation", fake_generate_estimation
    )
    monkeypatch.setattr("app.api.sessions.format_response", fake_format_response)
    monkeypatch.setattr(
        "app.api.sessions.extract_project_metadata",
        fake_extract_project_metadata,
    )

    output_path = tmp_path / "results.csv"

    await run_stress(
        http_base_url=None,
        scenario_names=["growing"],
        attachment_sizes_kb=[0],
        repeats=1,
        output_path=output_path,
        turn_counts=[3],
        latency_budget_ms=4000,
        cost_budget_usd=0.05,
    )

    rows = list(csv.DictReader(output_path.open()))

    assert len(rows) == 3
    assert rows[0]["scenario"] == "growing_3"
    assert rows[0]["attachment_size_kb"] == "0"
    assert rows[0]["turn_index"] == "1"
    assert rows[0]["latency_ms"] == "123"
    assert rows[0]["latency_budget_passed"] == "1"
    assert rows[0]["cost_budget_passed"] == "1"
    assert rows[0]["memory_drift_passed"] == "1"
    assert rows[2]["turn_index"] == "3"
