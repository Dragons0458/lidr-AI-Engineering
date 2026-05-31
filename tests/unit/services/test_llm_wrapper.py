from types import SimpleNamespace
from unittest.mock import MagicMock

import fakeredis
import pytest

from app.services.cache import EstimationCache
from app.services.llm_wrapper import LLMWrapper


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

    monkeypatch.setattr("app.services.llm_wrapper.completion", fake_completion)
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
        "app.services.llm_wrapper.completion",
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

    monkeypatch.setattr("app.services.llm_wrapper.completion", fake_completion)
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

    monkeypatch.setattr("app.services.llm_wrapper.log.warning", fake_warning)
    monkeypatch.setattr(
        "app.services.llm_wrapper.completion",
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


def test_complete_stream_replays_cache(monkeypatch, wrapper: LLMWrapper) -> None:
    monkeypatch.setattr(
        "app.services.llm_wrapper.completion",
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
