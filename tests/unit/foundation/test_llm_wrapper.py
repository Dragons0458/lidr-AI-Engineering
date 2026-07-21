from types import SimpleNamespace
from unittest.mock import MagicMock

import fakeredis
import pytest

from pydantic import BaseModel, Field

from instructor import Mode

from app.generation.cag.exact import EstimationCache
from app.foundation.llm.wrapper import LLMWrapper, structured_instructor_mode


class _SampleStructured(BaseModel):
    answer: str = Field(min_length=1)


def _llm_response(
    content: str,
    *,
    model: str = "gpt-4o-mini",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content),
            )
        ],
    )


@pytest.fixture
def wrapper() -> LLMWrapper:
    cache = EstimationCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)
    return LLMWrapper(
        primary_model="gpt-4o-mini",
        fallback_model=None,
        timeout=30,
        num_retries=0,
        cache=cache,
        cache_enabled=True,
    )


def test_complete_normalizes_and_caches(monkeypatch, wrapper: LLMWrapper) -> None:
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _llm_response("estimation text")

    monkeypatch.setattr("app.foundation.llm.wrapper.completion", fake_completion)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]

    first = wrapper.complete(
        messages=messages, model="gpt-4o-mini", max_tokens=1000, thinking_budget=None
    )
    second = wrapper.complete(
        messages=messages, model="gpt-4o-mini", max_tokens=1000, thinking_budget=None
    )

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert len(calls) == 1
    assert first["estimation"] == "estimation text"
    assert first["cost_usd"] > 0


def test_complete_unknown_model_has_zero_cost(monkeypatch, wrapper: LLMWrapper) -> None:
    monkeypatch.setattr(
        "app.foundation.llm.wrapper.completion",
        lambda **kwargs: _llm_response("x", model="unknown-model-xyz"),
    )
    result = wrapper.complete(
        messages=[{"role": "user", "content": "u"}],
        model="unknown-model-xyz",
        max_tokens=100,
    )
    assert result["cost_usd"] == 0.0


def test_thinking_budget_for_anthropic(monkeypatch, wrapper: LLMWrapper) -> None:
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _llm_response("x", model="claude-haiku-4-5-20251001")

    monkeypatch.setattr("app.foundation.llm.wrapper.completion", fake_completion)
    wrapper.complete(
        messages=[{"role": "user", "content": "u"}],
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        thinking_budget=2000,
    )
    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    assert captured["max_tokens"] == 2000 + 1024


def test_thinking_budget_warning_for_openai(monkeypatch, wrapper: LLMWrapper) -> None:
    warned: list[str] = []

    def fake_warning(event: str, **kwargs) -> None:
        warned.append(event)

    monkeypatch.setattr("app.foundation.llm.wrapper.log.warning", fake_warning)
    monkeypatch.setattr(
        "app.foundation.llm.wrapper.completion",
        lambda **kwargs: _llm_response("x"),
    )
    wrapper.complete(
        messages=[{"role": "user", "content": "u"}],
        model="gpt-4o-mini",
        max_tokens=1000,
        thinking_budget=2000,
    )
    assert warned == ["thinking_budget_ignored_for_provider"]


def test_fallback_uses_router(monkeypatch) -> None:
    cache = EstimationCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)
    wrapper = LLMWrapper(
        primary_model="gpt-4o-mini",
        fallback_model="claude-haiku-4-5-20251001",
        timeout=30,
        num_retries=0,
        cache=cache,
    )
    router_mock = MagicMock()
    router_mock.completion.return_value = _llm_response("from router")
    wrapper.router = router_mock

    result = wrapper.complete(
        messages=[{"role": "user", "content": "u"}],
        model="gpt-4o-mini",
        max_tokens=500,
    )
    router_mock.completion.assert_called_once()
    assert result["estimation"] == "from router"


def test_structured_instructor_mode_uses_json_for_google_litellm() -> None:
    assert structured_instructor_mode("gemini-2.5-flash") == Mode.JSON
    assert structured_instructor_mode("gemini/gemini-2.5-flash") == Mode.JSON


def test_structured_instructor_mode_uses_tools_for_openai() -> None:
    assert structured_instructor_mode("gpt-4o-mini") == Mode.TOOLS


def test_complete_structured_chat_disables_gemini_thinking(
    monkeypatch, wrapper: LLMWrapper
) -> None:
    captured: dict = {}

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    return _SampleStructured(answer="ok")

    monkeypatch.setattr(
        "app.foundation.llm.wrapper._get_instructor_client",
        lambda _model: FakeClient(),
    )
    wrapper.complete_structured_chat(
        messages=[{"role": "user", "content": "hi"}],
        response_model=_SampleStructured,
        model="gemini/gemini-2.5-flash",
    )
    assert captured["reasoning_effort"] == "none"


def test_complete_structured_chat_returns_model_and_meta(
    monkeypatch, wrapper: LLMWrapper
) -> None:
    expected = _SampleStructured(answer="structured")

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return expected

    monkeypatch.setattr(
        "app.foundation.llm.wrapper._get_instructor_client",
        lambda _model: FakeClient(),
    )
    instance, meta = wrapper.complete_structured_chat(
        messages=[{"role": "user", "content": "hi"}],
        response_model=_SampleStructured,
        model="gpt-4o-mini",
    )
    assert instance.answer == "structured"
    assert meta["model"] == "gpt-4o-mini"
    assert meta["cache_hit"] is False


def test_complete_structured_omits_unsupported_temperature_for_gpt5(
    monkeypatch, wrapper: LLMWrapper
) -> None:
    captured: dict = {}

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    return _SampleStructured(answer="ok")

    monkeypatch.setattr(
        "app.foundation.llm.wrapper._get_instructor_client",
        lambda _model: FakeClient(),
    )
    wrapper.complete_structured(
        system_prompt="route",
        user_message="state",
        response_model=_SampleStructured,
        model_override="gpt-5-mini",
    )

    assert "temperature" not in captured
    assert captured["reasoning_effort"] == "none"


def test_complete_structured_keeps_temperature_for_non_reasoning_model(
    monkeypatch, wrapper: LLMWrapper
) -> None:
    captured: dict = {}

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    return _SampleStructured(answer="ok")

    monkeypatch.setattr(
        "app.foundation.llm.wrapper._get_instructor_client",
        lambda _model: FakeClient(),
    )
    wrapper.complete_structured(
        system_prompt="route",
        user_message="state",
        response_model=_SampleStructured,
        model_override="gpt-4o-mini",
    )

    assert captured["temperature"] == 0
    assert "reasoning_effort" not in captured


def test_complete_stream_replays_cache(monkeypatch, wrapper: LLMWrapper) -> None:
    monkeypatch.setattr(
        "app.foundation.llm.wrapper.completion",
        lambda **kwargs: iter(
            [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))]
                )
            ]
        ),
    )
    messages = [{"role": "user", "content": "u"}]
    first = list(
        wrapper.complete_stream(messages=messages, model="gpt-4o-mini", max_tokens=500)
    )
    second = list(
        wrapper.complete_stream(messages=messages, model="gpt-4o-mini", max_tokens=500)
    )
    assert first == ["Hi"]
    assert second == ["Hi"]
