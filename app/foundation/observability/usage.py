"""Per-request LLM usage accumulator (Session 16).

A single HTTP estimate fans out into several LLM calls (reformulation,
generation, citation retry, hallucination judge, graph agents…). Cost is
recorded in ``LLMWrapper`` — the only choke point — by mutating a
``RequestUsage`` object stored in a ``ContextVar``.

The object is **mutable on purpose**. ``asyncio.to_thread`` copies the
context; rebinding the ContextVar inside the worker is invisible to the
caller. Mutating the same dataclass the middleware created is visible.

Never store transcripts, prompts, completions, or API keys here.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class RequestUsage:
    """Mutable accumulator shared across threads of one HTTP request."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    models: set[str] = field(default_factory=set)
    cache_hits: int = 0
    outcome_confidence: str | None = None
    outcome_abstained: bool | None = None
    outcome_grounded_ratio: float | None = None


_current_usage: ContextVar[RequestUsage | None] = ContextVar(
    "s16_request_usage", default=None
)


def start_usage() -> RequestUsage:
    """Create the accumulator for this request and bind it to the context."""
    usage = RequestUsage()
    _current_usage.set(usage)
    return usage


def get_usage() -> RequestUsage | None:
    """Return the current accumulator, or ``None`` outside a request."""
    return _current_usage.get()


def add_llm_call(
    *,
    model: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
    cache_hit: bool = False,
) -> None:
    """Fold one LLM call into the current request. No-op without a context."""
    usage = get_usage()
    if usage is None:
        return
    usage.llm_calls += 1
    usage.prompt_tokens += int(prompt_tokens or 0)
    usage.completion_tokens += int(completion_tokens or 0)
    usage.cost_usd += float(cost_usd or 0.0)
    if model:
        usage.models.add(str(model))
    if cache_hit:
        usage.cache_hits += 1


def set_outcome(
    *,
    confidence: str | None = None,
    abstained: bool | None = None,
    grounded_ratio: float | None = None,
) -> None:
    """Record the estimate-level outcome the middleware cannot re-parse."""
    usage = get_usage()
    if usage is None:
        return
    if confidence is not None:
        usage.outcome_confidence = confidence
    if abstained is not None:
        usage.outcome_abstained = abstained
    if grounded_ratio is not None:
        usage.outcome_grounded_ratio = grounded_ratio
