"""The usage accumulator mutates a shared object so asyncio.to_thread is visible."""

from __future__ import annotations

import asyncio

from app.foundation.observability.usage import (
    add_llm_call,
    get_usage,
    set_outcome,
    start_usage,
)


def test_accumulator_sums_several_calls() -> None:
    usage = start_usage()
    add_llm_call(
        model="gpt-4o-mini", prompt_tokens=10, completion_tokens=4, cost_usd=0.01
    )
    add_llm_call(model="gpt-4o", prompt_tokens=20, completion_tokens=6, cost_usd=0.02)
    assert usage.llm_calls == 2
    assert usage.prompt_tokens == 30
    assert usage.completion_tokens == 10
    assert abs(usage.cost_usd - 0.03) < 1e-9
    assert usage.models == {"gpt-4o-mini", "gpt-4o"}


def test_without_context_does_not_raise() -> None:
    # Ensure we are not inside a leftover context from another test.
    start_usage()
    from app.foundation.observability import usage as usage_mod

    usage_mod._current_usage.set(None)
    add_llm_call(model="gpt-4o-mini", prompt_tokens=1, cost_usd=0.1)
    set_outcome(confidence="high", abstained=False)
    assert get_usage() is None


async def test_survives_asyncio_to_thread() -> None:
    usage = start_usage()

    def _in_thread() -> None:
        add_llm_call(
            model="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=40,
            cost_usd=0.005,
        )

    await asyncio.to_thread(_in_thread)
    assert usage.llm_calls == 1
    assert usage.prompt_tokens == 100
    assert get_usage() is usage
    assert get_usage().cost_usd == 0.005
