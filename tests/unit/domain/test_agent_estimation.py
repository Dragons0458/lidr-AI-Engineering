"""Unit tests for the Session 12 hybrid-agent conductor."""

import pytest

import app.domain.agent_estimation as ae
from app.domain.schemas.agent_trace import AgentTrace
from app.generation.agentic.agent_schemas import (
    AgentModuleNode,
    AgentRecoveryNeighbor,
    AgentStructure,
    AgentTaskDerivation,
    AgentTaskHoursRun,
    AgentTaskNode,
)
from app.generation.rag.schemas import (
    EstimationQuery,
    HourRange,
    TaskHoursEstimate,
    TaskHoursModuleInput,
    TaskHoursResult,
    TaskNeighbor,
)


def _modules(duplicate: bool = False):
    tasks = [{"name": "Build", "description": "first"}]
    if duplicate:
        tasks.append({"name": "Build", "description": "second"})
    return [TaskHoursModuleInput(name="Core", tasks=tasks)]


def _deterministic(*tasks: TaskHoursEstimate) -> TaskHoursResult:
    return TaskHoursResult(tasks=list(tasks))


def _task(**overrides) -> TaskHoursEstimate:
    values = {
        "module": "Core",
        "task": "Build",
        "estimated_hours": 30,
        "reliability": 0.8,
        "dispersion": 0.1,
        "has_match": True,
        "neighbors": [
            TaskNeighbor(
                source_id=1,
                budget_id="old",
                estimated_hours=30,
                distance=0.2,
            )
        ],
    }
    values.update(overrides)
    return TaskHoursEstimate(**values)


def _derivation(ref="task-0", **overrides) -> AgentTaskDerivation:
    values = {
        "task_ref": ref,
        "module": "Core",
        "task": "Build",
        "estimated_hours": 44,
        "reliability": 0.91,
        "dispersion": 0.07,
        "has_match": True,
        "neighbors": [
            AgentRecoveryNeighbor(
                source_id=9,
                budget_id="new",
                estimated_hours=44,
                distance=0.05,
            )
        ],
    }
    values.update(overrides)
    return AgentTaskDerivation(**values)


async def _run(
    monkeypatch, deterministic, derivations=(), *, modules=None, client=object()
):
    async def fake_estimate_all(*args, **kwargs):
        return deterministic

    monkeypatch.setattr(ae, "estimate_all", fake_estimate_all)

    async def fake_run(flagged, **kwargs):
        fake_run.flagged = flagged
        return AgentTaskHoursRun(
            derivations=list(derivations), trace=AgentTrace(), iterations=1
        )

    monkeypatch.setattr(ae, "run_task_hours_recovery_agent", fake_run)
    monkeypatch.setattr(ae, "make_retrieval_backend", lambda **kwargs: object())
    return await ae.agent_estimate_task_hours(
        modules or _modules(),
        client=client,
        model="gpt-5",
        reasoning_effort="medium",
        max_iterations=10,
        top_k=5,
        distance_threshold=0.45,
        search_mode="hybrid",
        rerank=True,
        persona=None,
        recovery_reliability_threshold=0.35,
    )


@pytest.mark.asyncio
async def test_structure_maps_without_hours_or_sources(monkeypatch):
    structure = AgentStructure(
        modules=[
            AgentModuleNode(
                name="Auth",
                description="Identity",
                tasks=[AgentTaskNode(name="Login", description="OIDC")],
            )
        ],
        confidence="high",
        reasoning="Clear scope",
    )

    async def fake_structure(*args, **kwargs):
        return structure, AgentTrace()

    monkeypatch.setattr(ae, "run_structure_agent", fake_structure)
    result = await ae.agent_propose_structure(
        EstimationQuery(function="Portal"),
        client=object(),
        model="gpt-5",
        reasoning_effort="medium",
        persona=None,
    )
    task = result.estimate.modules[0].tasks[0]
    assert task.engineer_days is None
    assert task.grounded is False
    assert task.sources == []
    assert result.estimate.sources == []
    assert result.fabricated_source_ids == []
    assert result.coherent is True
    assert result.agent_trace is not None


@pytest.mark.asyncio
async def test_empty_structure_becomes_insufficient(monkeypatch):
    async def fake_structure(*args, **kwargs):
        return (
            AgentStructure(
                modules=[],
                confidence="low",
                reasoning="Missing scope",
                insufficient_context_explanation="Need requirements",
            ),
            AgentTrace(),
        )

    monkeypatch.setattr(ae, "run_structure_agent", fake_structure)
    result = await ae.agent_propose_structure(
        EstimationQuery(function="Unknown"),
        client=object(),
        model="gpt-5",
        reasoning_effort="medium",
        persona=None,
    )
    assert result.estimate.confidence == "insufficient"
    assert result.estimate.insufficient_context_explanation == "Need requirements"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "estimate, expected_reason",
    [
        (
            _task(
                has_match=False, estimated_hours=None, reliability=None, neighbors=[]
            ),
            "no historical match",
        ),
        (
            _task(hours_range=HourRange(low=10, high=80, reason="contradiction")),
            "contradictory historical range",
        ),
        (_task(reliability=0.34), "reliability below 0.35"),
    ],
)
async def test_all_recovery_flags(monkeypatch, estimate, expected_reason):
    captured = {}

    async def estimate_all(*args, **kwargs):
        return _deterministic(estimate)

    async def fake_run(flagged, **kwargs):
        captured["flagged"] = flagged
        return AgentTaskHoursRun(iterations=1)

    monkeypatch.setattr(ae, "estimate_all", estimate_all)
    monkeypatch.setattr(ae, "run_task_hours_recovery_agent", fake_run)
    monkeypatch.setattr(ae, "make_retrieval_backend", lambda **kwargs: object())
    await ae.agent_estimate_task_hours(
        _modules(),
        client=object(),
        model="gpt-5",
        reasoning_effort="medium",
        max_iterations=10,
        top_k=5,
        distance_threshold=0.45,
        search_mode="vector",
        rerank=False,
        persona=None,
        recovery_reliability_threshold=0.35,
    )
    assert expected_reason in captured["flagged"][0].reason


@pytest.mark.asyncio
async def test_boundary_and_clean_results_short_circuit_without_client(monkeypatch):
    deterministic = _deterministic(_task(reliability=0.35))

    async def estimate_all(*args, **kwargs):
        return deterministic

    monkeypatch.setattr(ae, "estimate_all", estimate_all)
    result = await ae.agent_estimate_task_hours(
        _modules(),
        client=None,
        model="gpt-5",
        reasoning_effort="medium",
        max_iterations=10,
        top_k=5,
        distance_threshold=0.45,
        search_mode="vector",
        rerank=False,
        persona=None,
        recovery_reliability_threshold=0.35,
    )
    assert result.tasks == deterministic.tasks
    assert result.agent_trace is not None and result.agent_trace.steps == []


@pytest.mark.asyncio
async def test_client_is_required_only_when_flagged(monkeypatch):
    async def estimate_all(*args, **kwargs):
        return _deterministic(
            _task(has_match=False, estimated_hours=None, reliability=None, neighbors=[])
        )

    monkeypatch.setattr(ae, "estimate_all", estimate_all)
    with pytest.raises(ae.OpenAIClientMissingError):
        await ae.agent_estimate_task_hours(
            _modules(),
            client=None,
            model="gpt-5",
            reasoning_effort="medium",
            max_iterations=10,
            top_k=5,
            distance_threshold=0.45,
            search_mode="vector",
            rerank=False,
            persona=None,
            recovery_reliability_threshold=0.35,
        )


@pytest.mark.asyncio
async def test_merge_by_ref_replaces_provenance_and_clears_only_replaced_range(
    monkeypatch,
):
    flagged = _task(
        reliability=0.2,
        hours_range=HourRange(low=10, high=70, reason="wide"),
    )
    result = await _run(monkeypatch, _deterministic(flagged), [_derivation()])
    merged = result.tasks[0]
    assert merged.estimated_hours == 44
    assert merged.reliability == 0.91
    assert merged.dispersion == 0.07
    assert merged.neighbors[0].source_id == 9
    assert merged.neighbors[0].budget_id == "new"
    assert merged.hours_range is None
    assert merged.estimation_source == "agent_recovery"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "derivation",
    [
        _derivation("unknown"),
        _derivation(module="Changed"),
        _derivation(task="Changed"),
    ],
)
async def test_rejects_unknown_refs_and_changed_identity(monkeypatch, derivation):
    with pytest.raises(ae.RecoveryAgentError):
        await _run(monkeypatch, _deterministic(_task(reliability=0.2)), [derivation])


@pytest.mark.asyncio
async def test_duplicate_names_recover_independently(monkeypatch):
    deterministic = _deterministic(
        _task(has_match=False, estimated_hours=None, reliability=None, neighbors=[]),
        _task(has_match=False, estimated_hours=None, reliability=None, neighbors=[]),
    )
    result = await _run(
        monkeypatch,
        deterministic,
        [_derivation("task-0"), _derivation("task-1", estimated_hours=55)],
        modules=_modules(duplicate=True),
    )
    assert [task.estimated_hours for task in result.tasks] == [44, 55]


@pytest.mark.asyncio
async def test_incomplete_derivation_does_not_overwrite_and_unrecovered_stays_unresolved(
    monkeypatch,
):
    unresolved = _task(
        has_match=False, estimated_hours=None, reliability=None, neighbors=[]
    )
    incomplete = AgentTaskDerivation(
        task_ref="task-0",
        module="Core",
        task="Build",
        has_match=False,
    )
    result = await _run(monkeypatch, _deterministic(unresolved), [incomplete])
    assert result.tasks[0] == unresolved
    assert result.tasks[0].estimated_hours is None
