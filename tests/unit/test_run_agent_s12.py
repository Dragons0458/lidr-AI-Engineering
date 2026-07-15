"""Unit tests for the Session 12 CLI workflow selection."""

import argparse
from types import SimpleNamespace

import pytest

import scripts.run_agent_s12 as cli
from app.domain.schemas.agent_trace import AgentTrace
from app.generation.agentic.agent_schemas import (
    AgentEstimate,
    AgentRunResult,
    AgentStructure,
    AgentTaskHoursRun,
)
from app.generation.rag.schemas import (
    Estimate,
    EstimationQuery,
    GenerateResult,
    TaskHoursResult,
)


def _args(path, workflow, *, stub=False):
    return argparse.Namespace(
        transcript=str(path),
        workflow=workflow,
        model="gpt-5-mini",
        effort="low",
        max_iterations=4,
        stub=stub,
        out=None,
        persona="Careful",
    )


def _settings():
    return SimpleNamespace(
        TASK_HOURS_TOP_K=5,
        TASK_HOURS_DISTANCE_THRESHOLD=0.45,
        RETRIEVAL_SEARCH_MODE="hybrid",
        RERANKER_ENABLED=True,
        AGENT_RECOVERY_RELIABILITY_THRESHOLD=0.35,
        AGENT_SEARCH_TOP_K=5,
        AGENT_SEARCH_DISTANCE_THRESHOLD=0.45,
        AGENT_MODEL="gpt-5",
        AGENT_REASONING_EFFORT="medium",
        AGENT_MAX_ITERATIONS=10,
    )


@pytest.fixture
def transcript(tmp_path):
    path = tmp_path / "meeting.txt"
    path.write_text("Build a portal with authentication.", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setattr(cli, "get_async_openai_client", lambda: object())
    monkeypatch.setattr(cli, "get_settings", _settings)

    async def reformulate(_transcript):
        return EstimationQuery(function="Portal")

    monkeypatch.setattr(cli, "reformulate_query", reformulate)


@pytest.mark.asyncio
async def test_hybrid_uses_shared_conductor(monkeypatch, transcript):
    calls = []

    async def structure(*args, **kwargs):
        calls.append(("structure", kwargs))
        return GenerateResult(
            estimate=Estimate(
                confidence="high",
                reasoning="clear",
                modules=[{"name": "Core", "tasks": [{"name": "Build"}]}],
            ),
            coherent=True,
            agent_trace=AgentTrace(),
        )

    async def hours(modules, **kwargs):
        calls.append(("hours", kwargs))
        return TaskHoursResult(agent_trace=AgentTrace())

    monkeypatch.setattr(cli, "agent_propose_structure", structure)
    monkeypatch.setattr(cli, "agent_estimate_task_hours", hours)
    assert await cli._main_async(_args(transcript, "hybrid")) == 0
    assert [name for name, _ in calls] == ["structure", "hours"]
    assert calls[1][1]["recovery_reliability_threshold"] == 0.35


@pytest.mark.asyncio
async def test_recovery_demo_flags_every_structure_task(monkeypatch, transcript):
    async def structure(*args, **kwargs):
        return (
            AgentStructure(
                modules=[
                    {
                        "name": "Core",
                        "tasks": [{"name": "A"}, {"name": "B"}],
                    }
                ],
                confidence="high",
                reasoning="clear",
            ),
            AgentTrace(),
        )

    captured = {}

    async def recovery(flagged, **kwargs):
        captured["flagged"] = flagged
        return AgentTaskHoursRun(iterations=1)

    monkeypatch.setattr(cli, "run_structure_agent", structure)
    monkeypatch.setattr(cli, "run_task_hours_recovery_agent", recovery)
    monkeypatch.setattr(cli, "make_retrieval_backend", lambda **kwargs: object())
    assert await cli._main_async(_args(transcript, "recovery-demo")) == 0
    assert [task.task_ref for task in captured["flagged"]] == ["task-0", "task-1"]
    assert all(task.reason == "recovery demo" for task in captured["flagged"])


@pytest.mark.asyncio
async def test_legacy_keeps_one_shot_agent(monkeypatch, transcript):
    captured = {}

    async def legacy(text, **kwargs):
        captured["text"] = text
        return AgentRunResult(
            estimate=AgentEstimate(
                components=[],
                total_hours=0,
                confidence="low",
            ),
            trace=AgentTrace(),
            iterations=1,
        )

    monkeypatch.setattr(cli, "run_estimation_agent", legacy)
    assert await cli._main_async(_args(transcript, "legacy")) == 0
    assert "authentication" in captured["text"]


@pytest.mark.asyncio
async def test_stub_adapters_work_for_legacy_and_recovery(monkeypatch):
    async def legacy_stub(args):
        return [{"query": args.query, "sectors": args.filters.sectors}]

    monkeypatch.setattr(cli, "_load_stub_backend", lambda: legacy_stub)
    legacy = cli._load_stub_backend()
    recovery = cli._load_recovery_stub_backend()
    from app.generation.agentic.agent_schemas import SearchBudgetsArgs

    args = SearchBudgetsArgs(
        query="auth",
        filters={"sectors": ["finance"], "component_type": None},
    )
    assert (await legacy(args))[0]["query"] == "auth"
    assert (await recovery("auth", ["finance"]))[0]["sectors"] == ["finance"]


def test_stub_with_hybrid_is_usage_error(monkeypatch, transcript):
    monkeypatch.setattr(
        "sys.argv",
        [str(cli.__file__), str(transcript), "--workflow", "hybrid", "--stub"],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


@pytest.mark.asyncio
async def test_missing_transcript_or_client_returns_nonzero(
    monkeypatch, transcript, tmp_path
):
    assert await cli._main_async(_args(tmp_path / "missing.txt", "legacy")) != 0
    monkeypatch.setattr(cli, "get_async_openai_client", lambda: None)
    assert await cli._main_async(_args(transcript, "legacy")) != 0
