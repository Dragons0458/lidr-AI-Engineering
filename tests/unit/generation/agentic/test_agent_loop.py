"""Unit tests for the manual agent loop, driven by a fake AsyncOpenAI client.

No network and no API key: a scripted fake client returns canned Responses API
outputs so we can assert the loop's control flow — multiple tool calls, call_id
echoing, the max-iterations safeguard and the trace shape.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from app.generation.agentic.agent_loop import run_estimation_agent
from app.generation.agentic.agent_schemas import (
    AgentComponent,
    AgentEstimate,
    AgentModuleNode,
    AgentStructure,
    AgentTaskNode,
    AgentTaskRef,
    SearchBudgetsArgs,
)
from app.generation.agentic.agent_loop import (
    run_structure_agent,
    run_task_hours_recovery_agent,
)


def _function_call(name: str, call_id: str, arguments: dict):
    return SimpleNamespace(
        type="function_call",
        name=name,
        call_id=call_id,
        arguments=json.dumps(arguments),
    )


def _reasoning(text: str):
    return SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text=text)])


def _message():
    return SimpleNamespace(type="message", role="assistant", content=[])


class _FakeResponses:
    """Scripted ``responses.create`` / ``responses.parse`` double."""

    def __init__(self, scripted_outputs: list[list], parsed: AgentEstimate | None):
        self._scripted = scripted_outputs
        self._parsed = parsed
        self._i = 0
        self.create_calls: list[dict] = []
        self.parse_calls: list[dict] = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        output = self._scripted[min(self._i, len(self._scripted) - 1)]
        self._i += 1
        return SimpleNamespace(output=output, id=f"resp_{self._i}")

    async def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        return SimpleNamespace(output_parsed=self._parsed, output=[])


class _FakeClient:
    def __init__(self, responses: _FakeResponses):
        self.responses = responses


async def _stub_backend(args: SearchBudgetsArgs) -> list[dict]:
    return [
        {"id": 1, "estimated_hours": 100.0, "content_preview": "x", "distance": 0.1}
    ]


def _happy_path_script():
    """Two searches → calculate → validate → final message (loop ends)."""
    return [
        [
            _reasoning("Decompose the project and search each component."),
            _function_call(
                "search_budgets", "call_1", {"query": "auth backend", "filters": None}
            ),
            _function_call(
                "search_budgets", "call_2", {"query": "mobile app", "filters": None}
            ),
        ],
        [
            _function_call(
                "calculate_estimate",
                "call_3",
                {"components": [{"name": "Auth", "reference_amounts": [100.0]}]},
            )
        ],
        [
            _function_call(
                "validate_estimate",
                "call_4",
                {
                    "components": [
                        {
                            "name": "Auth",
                            "estimated_hours": 115.0,
                            "reference_amounts": [100.0],
                        }
                    ],
                    "total_hours": 115.0,
                },
            )
        ],
        [_message()],
    ]


def _final_estimate() -> AgentEstimate:
    return AgentEstimate(
        components=[
            AgentComponent(
                name="Auth", estimated_hours=115.0, rationale="median+buffer"
            )
        ],
        total_hours=115.0,
        assumptions=["Rails/Postgres as stated."],
        confidence="medium",
    )


async def test_happy_path_multi_tool_run():
    fake = _FakeResponses(_happy_path_script(), _final_estimate())
    result = await run_estimation_agent(
        "transcript text",
        client=_FakeClient(fake),
        model="gpt-5-mini",
        max_iterations=10,
        retrieval_backend=_stub_backend,
    )

    tools_used = [step.tool for step in result.trace.steps]
    assert (
        tools_used.count("search_budgets") == 2
    )  # >1 search, per the acceptance criteria
    assert "calculate_estimate" in tools_used
    assert "validate_estimate" in tools_used

    # Every step carries reasoning + action + observation.
    for step in result.trace.steps:
        assert step.tool
        assert step.observation
    assert result.trace.steps[0].reasoning_summary is not None

    assert result.stopped_reason == "completed"
    assert result.estimate is not None
    assert result.estimate.total_hours == 115.0
    assert fake.parse_calls, "final structured parse should have been called"


async def test_call_ids_are_echoed_back():
    fake = _FakeResponses(_happy_path_script(), _final_estimate())
    await run_estimation_agent(
        "t",
        client=_FakeClient(fake),
        model="gpt-5-mini",
        retrieval_backend=_stub_backend,
    )
    # The 2nd create call carries the outputs for the first turn's two calls.
    second_call_input = fake.create_calls[1]["input"]
    echoed = {item["call_id"] for item in second_call_input}
    assert echoed == {"call_1", "call_2"}
    for item in second_call_input:
        assert item["type"] == "function_call_output"
        assert isinstance(item["output"], str)  # output must be a JSON string


async def test_max_iterations_safeguard_stops_loop():
    # A script that never stops calling a tool.
    never_stops = [
        [_function_call("search_budgets", "call_x", {"query": "loop", "filters": None})]
    ]
    fake = _FakeResponses(never_stops, _final_estimate())
    result = await run_estimation_agent(
        "t",
        client=_FakeClient(fake),
        model="gpt-5-mini",
        max_iterations=3,
        retrieval_backend=_stub_backend,
    )
    assert result.stopped_reason == "max_iterations"
    assert result.estimate is None
    assert not fake.parse_calls  # no final parse when the safeguard trips
    assert result.iterations == 3


async def test_bad_tool_arguments_do_not_crash_the_loop():
    # First turn calls calculate_estimate with structurally invalid args, then stops.
    script = [
        [
            _function_call(
                "calculate_estimate", "call_1", {"components": [{"name": "A"}]}
            )
        ],
        [_message()],
    ]
    fake = _FakeResponses(script, _final_estimate())
    result = await run_estimation_agent(
        "t",
        client=_FakeClient(fake),
        model="gpt-5-mini",
        retrieval_backend=_stub_backend,
    )
    # The bad call becomes an error observation, not an exception.
    assert result.trace.steps[0].tool == "calculate_estimate"
    assert "error" in result.trace.steps[0].observation.lower()


def _structure() -> AgentStructure:
    return AgentStructure(
        modules=[
            AgentModuleNode(
                name="Core",
                tasks=[AgentTaskNode(name="Build", description="Implementation")],
            )
        ],
        confidence="high",
        reasoning="Clear scope",
    )


def _flagged() -> list[AgentTaskRef]:
    return [
        AgentTaskRef(
            task_ref="task-0",
            module="Core",
            task="Build",
            reason="no historical match",
        )
    ]


async def _recovery_backend(query, sectors):
    return [
        {
            "id": 7,
            "source_id": 7,
            "budget_id": "BUD-7",
            "estimated_hours": 40,
            "distance": 0.08,
        }
    ]


def _consensus(neighbors):
    return 40, 0.9, 0.0


async def test_structure_is_single_parse_without_tools_and_includes_persona():
    fake = _FakeResponses([], _structure())
    structure, trace = await run_structure_agent(
        {"function": "Portal"},
        client=_FakeClient(fake),
        model="gpt-5",
        reasoning_effort="medium",
        persona="Challenge assumptions",
    )
    call = fake.parse_calls[0]
    assert "tools" not in call
    assert call["text_format"] is AgentStructure
    assert "Challenge assumptions" in call["instructions"]
    assert structure.modules[0].tasks[0].name == "Build"
    assert trace.steps[0].tool == "propose_structure"
    assert "1 modules and 1 tasks" in trace.steps[0].observation


async def test_empty_recovery_returns_without_client_call():
    fake = _FakeResponses([], None)
    result = await run_task_hours_recovery_agent(
        [],
        client=_FakeClient(fake),
        model="gpt-5",
        reasoning_effort="medium",
        max_iterations=10,
        backend=_recovery_backend,
        consensus=_consensus,
    )
    assert result.iterations == 0
    assert result.derivations == []
    assert fake.create_calls == []


async def test_recovery_supports_reformulation_joint_outputs_chaining_and_provenance():
    neighbors = [
        {
            "source_id": 7,
            "budget_id": "BUD-7",
            "estimated_hours": 40,
            "distance": 0.08,
        }
    ]
    script = [
        [
            _reasoning("Search twice."),
            _function_call(
                "search_budgets",
                "search_1",
                {"query": "broad query", "sectors": None},
            ),
            _function_call(
                "search_budgets",
                "search_2",
                {"query": "focused reformulation", "sectors": ["finance"]},
            ),
        ],
        [
            _function_call(
                "derive_task_hours",
                "derive_1",
                {
                    "task_ref": "task-0",
                    "module": "Core",
                    "task": "Build",
                    "neighbors": neighbors,
                },
            )
        ],
        [_message()],
    ]
    fake = _FakeResponses(script, None)
    result = await run_task_hours_recovery_agent(
        _flagged(),
        client=_FakeClient(fake),
        model="gpt-5",
        reasoning_effort="medium",
        max_iterations=10,
        backend=_recovery_backend,
        consensus=_consensus,
    )
    assert [step.tool for step in result.trace.steps].count("search_budgets") == 2
    assert {item["call_id"] for item in fake.create_calls[1]["input"]} == {
        "search_1",
        "search_2",
    }
    assert fake.create_calls[1]["previous_response_id"] == "resp_1"
    assert result.derivations[0].neighbors[0].source_id == 7
    assert result.derivations[0].neighbors[0].budget_id == "BUD-7"
    assert result.derivations[0].dispersion == 0.0


async def test_recovery_max_iterations_and_recoverable_json_tool_errors():
    script = [
        [
            SimpleNamespace(
                type="function_call",
                name="search_budgets",
                call_id="bad",
                arguments="{",
            )
        ],
        [_function_call("unknown", "unknown", {})],
        [_function_call("search_budgets", "loop", {"query": "q", "sectors": None})],
    ]
    fake = _FakeResponses(script, None)
    result = await run_task_hours_recovery_agent(
        _flagged(),
        client=_FakeClient(fake),
        model="gpt-5",
        reasoning_effort="medium",
        max_iterations=3,
        backend=_recovery_backend,
        consensus=_consensus,
    )
    assert result.stopped_reason == "max_iterations"
    assert "valid JSON" in result.trace.steps[0].observation
    assert "Unknown recovery tool" in result.trace.steps[1].observation


class _SummaryFallbackResponses:
    def __init__(self, error):
        self.error = error
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise self.error
        return SimpleNamespace(output=[_message()], id="resp_ok")

    async def parse(self, **kwargs):
        return SimpleNamespace(output_parsed=_final_estimate(), output=[])


async def test_reasoning_summary_fallback_only_for_capability_error():
    responses = _SummaryFallbackResponses(
        RuntimeError("reasoning summary is not supported by this model")
    )
    result = await run_estimation_agent(
        "transcript",
        client=_FakeClient(responses),
        model="gpt-5",
        retrieval_backend=_stub_backend,
    )
    assert result.estimate is not None
    assert responses.calls[0]["reasoning"]["summary"] == "auto"
    assert "summary" not in responses.calls[1]["reasoning"]


async def test_other_openai_errors_propagate():
    responses = _SummaryFallbackResponses(RuntimeError("authentication failed"))
    try:
        await run_estimation_agent(
            "transcript",
            client=_FakeClient(responses),
            model="gpt-5",
            retrieval_backend=_stub_backend,
        )
    except RuntimeError as exc:
        assert "authentication" in str(exc)
    else:
        raise AssertionError("non-capability errors must propagate")
