"""LiteLLM-backed wrapper: fallback, exact-match cache, cost tracking, structured logs."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any, TypeVar

import instructor
import structlog
from instructor import Mode
from litellm import Router
from pydantic import BaseModel
from litellm import completion as litellm_completion

from app.foundation.llm.pricing import calculate_cost
from app.foundation.llm.runtime_config import RuntimeModelConfig
from app.generation.cag.exact import EstimationCache

log = structlog.get_logger()
T = TypeVar("T", bound=BaseModel)
_instructor_clients: dict[Mode, Any] = {}


def structured_instructor_mode(model: str) -> Mode:
    """Pick Instructor mode for structured output via LiteLLM.

    Google/Gemini: ``Mode.JSON`` (``response_format=json_object`` + schema in the
    system prompt). ``Mode.GEMINI_JSON`` is for the native Gemini SDK and forbids
    passing ``model`` on ``create()`` — incompatible with ``from_litellm``.

    OpenAI/Anthropic: tool calling via ``Mode.TOOLS``.
    """
    if provider_from_model(model) == "google":
        return Mode.JSON
    return Mode.TOOLS


def _get_instructor_client(model: str) -> Any:
    """Return a cached Instructor client for the mode required by ``model``."""
    mode = structured_instructor_mode(model)
    if mode not in _instructor_clients:
        _instructor_clients[mode] = instructor.from_litellm(completion, mode=mode)
    return _instructor_clients[mode]


def _normalise_model_name(model: str) -> str:
    return model.split("/", 1)[-1] if "/" in model else model


def provider_from_model(model: str) -> str:
    name = _normalise_model_name(model).lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith("gpt") or name.startswith("o1") or name.startswith("o3"):
        return "openai"
    if name.startswith("gemini"):
        return "google"
    return "unknown"


def completion(**kwargs: Any) -> Any:
    """Module-level seam for tests (patch ``app.foundation.llm.wrapper.completion``)."""
    return litellm_completion(**kwargs)


def _extract_delta(chunk: Any) -> str:
    try:
        delta = chunk.choices[0].delta
    except (AttributeError, IndexError):
        return ""
    content = getattr(delta, "content", None)
    return content or ""


class LLMWrapper:
    """Unified LLM client with cache, optional fallback, and cost tracking."""

    def __init__(
        self,
        *,
        primary_model: str,
        fallback_model: str | None,
        timeout: int,
        num_retries: int,
        cache: EstimationCache,
        cache_enabled: bool = True,
        runtime_config: RuntimeModelConfig | None = None,
    ):
        self._initial_primary_model = primary_model
        self._initial_fallback_model = fallback_model
        self._runtime_config = runtime_config
        self.timeout = timeout
        self.num_retries = num_retries
        self.cache = cache
        self.cache_enabled = cache_enabled
        self.router: Router | None = None

        if fallback_model:
            self.router = Router(
                model_list=[
                    {
                        "model_name": "estimator",
                        "litellm_params": {
                            "model": primary_model,
                            "timeout": timeout,
                        },
                    },
                    {
                        "model_name": "estimator_fallback",
                        "litellm_params": {
                            "model": fallback_model,
                            "timeout": timeout,
                        },
                    },
                ],
                fallbacks=[{"estimator": ["estimator_fallback"]}],
                num_retries=num_retries,
            )

    @property
    def primary_model(self) -> str:
        if self._runtime_config is not None:
            return self._runtime_config.effective("PRIMARY_MODEL")
        return self._initial_primary_model

    @property
    def fallback_model(self) -> str | None:
        if self._runtime_config is not None:
            value = self._runtime_config.effective("FALLBACK_MODEL")
            return value or None
        return self._initial_fallback_model

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        thinking_budget: int | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        cache_key = EstimationCache.make_key(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
        if self.cache_enabled and use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                return {**cached, "cache_hit": True}

        kwargs = self._build_call_kwargs(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )

        log.info(
            "llm_call_started",
            mode="blocking",
            model=model,
            has_thinking=thinking_budget is not None,
        )
        started_at = time.perf_counter()
        try:
            response = self._dispatch(model=model, **kwargs)
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            log.error(
                "llm_call_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                latency_ms=latency_ms,
            )
            raise

        latency_ms = int((time.perf_counter() - started_at) * 1000)
        result = self._normalise_response(response, latency_ms=latency_ms)
        resolved_model = result["model"]
        fallback_used = self.fallback_model is not None and _normalise_model_name(
            resolved_model
        ) == _normalise_model_name(self.fallback_model)
        log.info(
            "llm_call_completed",
            model=resolved_model,
            provider=result["provider"],
            input_tokens=result["usage"]["input_tokens"],
            output_tokens=result["usage"]["output_tokens"],
            cost_usd=result["cost_usd"],
            latency_ms=latency_ms,
            finish_reason=result["finish_reason"],
            fallback_used=fallback_used,
        )

        if self.cache_enabled and use_cache:
            self.cache.set(cache_key, result)

        return {**result, "cache_hit": False}

    def complete_structured_chat(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[T],
        model: str | None = None,
        max_tokens: int = 4000,
        max_retries: int = 2,
        temperature: float = 0,
        use_cache: bool = False,
    ) -> tuple[T, dict[str, Any]]:
        """Structured completion via Instructor; returns validated model and metadata."""
        resolved_model = model or self.primary_model
        instructor_mode = structured_instructor_mode(resolved_model)
        log.info(
            "llm_structured_call_started",
            model=resolved_model,
            response_model=response_model.__name__,
            instructor_mode=instructor_mode.value,
        )
        started_at = time.perf_counter()
        call_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "response_model": response_model,
            "max_tokens": max_tokens,
            "max_retries": max_retries,
            "timeout": self.timeout,
            "temperature": temperature,
        }
        provider = provider_from_model(resolved_model)
        # Gemini 2.5 counts hidden "thinking" tokens against max_output_tokens;
        # disable it for structured JSON so the budget goes to the visible response.
        if provider in ("openai", "google"):
            call_kwargs["reasoning_effort"] = "none"
        try:
            instance = _get_instructor_client(resolved_model).chat.completions.create(
                **call_kwargs
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            log.error(
                "llm_structured_call_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                latency_ms=latency_ms,
            )
            raise

        latency_ms = int((time.perf_counter() - started_at) * 1000)
        meta = {
            "model": _normalise_model_name(resolved_model),
            "provider": provider_from_model(resolved_model),
            "latency_ms": latency_ms,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "cost_usd": 0.0,
            "cache_hit": False,
        }
        log.info(
            "llm_structured_call_completed",
            model=meta["model"],
            latency_ms=latency_ms,
        )
        return instance, meta

    def complete_stream(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        thinking_budget: int | None = None,
        use_cache: bool = True,
    ) -> Iterator[str]:
        cache_key = EstimationCache.make_key(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
        if self.cache_enabled and use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                log.info(
                    "stream_cache_hit",
                    chars=len(cached.get("estimation", "")),
                )
                yield cached.get("estimation", "")
                return

        kwargs = self._build_call_kwargs(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            stream=True,
        )

        log.info("llm_stream_started", model=model)
        started_at = time.perf_counter()
        full_text: list[str] = []
        try:
            stream = self._dispatch(model=model, **kwargs)
            for chunk in stream:
                delta = _extract_delta(chunk)
                if delta:
                    full_text.append(delta)
                    yield delta
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            log.error(
                "llm_stream_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                latency_ms=latency_ms,
            )
            raise

        latency_ms = int((time.perf_counter() - started_at) * 1000)
        rendered = "".join(full_text)
        log.info("llm_stream_completed", latency_ms=latency_ms, chars=len(rendered))

        if self.cache_enabled and use_cache:
            self.cache.set(
                cache_key,
                {
                    "estimation": rendered,
                    "model": model,
                    "provider": provider_from_model(model),
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "latency_ms": latency_ms,
                    "cost_usd": 0.0,
                },
            )

    def _build_call_kwargs(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        thinking_budget: int | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if stream:
            kwargs["stream"] = True

        if thinking_budget is not None:
            if provider_from_model(model) == "anthropic":
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
                kwargs["max_tokens"] = max(max_tokens, thinking_budget + 1024)
            else:
                log.warning(
                    "thinking_budget_ignored_for_provider",
                    provider=provider_from_model(model),
                    model=model,
                )
                kwargs["reasoning_effort"] = "none"
        elif not stream:
            kwargs["reasoning_effort"] = "none"

        return kwargs

    def _dispatch(self, *, model: str, **kwargs: Any) -> Any:
        call_kwargs = {**kwargs, "timeout": self.timeout}
        if self.router and model == self.primary_model:
            return self.router.completion(
                model="estimator",
                num_retries=self.num_retries,
                **call_kwargs,
            )
        return completion(
            model=model,
            num_retries=self.num_retries,
            **call_kwargs,
        )

    @staticmethod
    def _normalise_response(response: Any, *, latency_ms: int) -> dict[str, Any]:
        choice = response.choices[0]
        finish_reason = (getattr(choice, "finish_reason", None) or "stop").lower()
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", input_tokens + output_tokens) or 0
        )

        resolved_model = _normalise_model_name(
            getattr(response, "model", "") or "unknown"
        )
        cost = calculate_cost(resolved_model, input_tokens, output_tokens)
        cost_usd = float(cost["total"]) if cost else 0.0

        return {
            "estimation": (choice.message.content or ""),
            "model": resolved_model,
            "provider": provider_from_model(resolved_model),
            "finish_reason": finish_reason,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
        }
