from types import SimpleNamespace

from app.formatters.llm_formatters import format_response
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)
from app.services.estimation_service import generate_estimation


def _request(**overrides) -> EstimationRequest:
    values = {
        "description": "Portal web con autenticacion, reportes y roles internos.",
        "project_type": ProjectType.WEB_SAAS,
        "detail_level": DetailLevel.MEDIUM,
        "output_format": OutputFormat.PHASES_TABLE,
    }
    values.update(overrides)
    return EstimationRequest(**values)


def _fake_response(content: str, input_tokens: int, output_tokens: int):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        ),
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content),
            )
        ],
    )


def test_two_phase_preprocessing_makes_two_calls_and_accumulates_usage(
    monkeypatch,
) -> None:
    calls = []

    def fake_call_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _fake_response("Requisitos extraidos", 5, 7)
        return _fake_response("## Estimacion\nTotal estimated hours: 20h", 11, 13)

    monkeypatch.setattr(
        "app.services.estimation_service._call_completion", fake_call_completion
    )

    result = generate_estimation(
        _request(preprocessing="two_phase", model="test-model", max_tokens=3000)
    )

    assert len(calls) == 2
    assert result.extracted_requirements == "Requisitos extraidos"
    assert result.input_tokens == 11
    assert result.output_tokens == 13
    assert result.preprocessing_input_tokens == 5
    assert result.preprocessing_output_tokens == 7

    formatted = format_response(result)
    assert formatted.usage.preprocessing_input_tokens == 5
    assert formatted.usage.preprocessing_output_tokens == 7
    assert formatted.usage.input_tokens == 11
    assert formatted.usage.output_tokens == 13
    assert formatted.usage.tokens_used == 36

    assert result.model == "test-model"
    assert calls[0]["max_tokens"] == 3000
    assert calls[1]["max_tokens"] == 3000
    assert "Requisitos extraidos" in calls[1]["messages"][-1]["content"]


def test_model_override_and_max_tokens_are_passed(monkeypatch) -> None:
    captured = {}

    def fake_call_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response("## Estimacion\nTotal estimated hours: 20h", 10, 12)

    monkeypatch.setattr(
        "app.services.estimation_service._call_completion", fake_call_completion
    )

    generate_estimation(_request(model="custom/model", max_tokens=2048))

    assert captured["model"] == "custom/model"
    assert captured["max_tokens"] == 2048


def test_use_examples_false_omits_examples_block(monkeypatch) -> None:
    captured = {}

    def fake_call_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response("## Estimacion\nTotal estimated hours: 20h", 10, 12)

    monkeypatch.setattr(
        "app.services.estimation_service._call_completion", fake_call_completion
    )

    generate_estimation(_request(use_examples=False))

    system_prompt = captured["messages"][0]["content"]
    assert "<examples>" not in system_prompt


def test_inline_cleaning_injects_block_without_second_call(monkeypatch) -> None:
    calls = []

    def fake_call_completion(**kwargs):
        calls.append(kwargs)
        return _fake_response("## Estimacion\nTotal estimated hours: 20h", 10, 12)

    monkeypatch.setattr(
        "app.services.estimation_service._call_completion", fake_call_completion
    )

    result = generate_estimation(_request(preprocessing="inline_cleaning"))

    assert len(calls) == 1
    assert result.extracted_requirements is None
    system_prompt = calls[0]["messages"][0]["content"]
    assert "<transcription_cleaning>" in system_prompt
    assert "Extract ONLY the functional and technical requirements" in system_prompt


def test_thinking_budget_passes_thinking_kwargs_for_anthropic(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.estimation_service.settings.LLM_PROVIDER", "anthropic"
    )
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response("## Estimacion\nTotal estimated hours: 20h", 10, 12)

    monkeypatch.setattr("app.services.estimation_service.completion", fake_completion)

    generate_estimation(_request(thinking_budget=2000, max_tokens=1000))

    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    assert captured["max_tokens"] == 2000 + 1024
    assert "reasoning_effort" not in captured


def test_thinking_budget_ignored_with_warning_for_openai(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.estimation_service.settings.LLM_PROVIDER", "openai"
    )
    warned: list[str] = []

    def fake_warning(event: str, **kwargs) -> None:
        warned.append(event)

    monkeypatch.setattr("app.services.estimation_service.log.warning", fake_warning)
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response("## Estimacion\nTotal estimated hours: 20h", 10, 12)

    monkeypatch.setattr("app.services.estimation_service.completion", fake_completion)

    generate_estimation(_request(thinking_budget=2000))

    assert warned == ["thinking_budget_ignored_for_provider"]
    assert "thinking" not in captured
    assert captured["reasoning_effort"] == "none"
