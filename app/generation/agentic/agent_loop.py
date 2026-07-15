"""The hand-written agent loop (Session 12).

A senior developer reads this and recognises everything: a loop that calls an LLM
which *decides*, runs *tools*, and stops when it is done. No framework.

DELIBERATE EXCEPTION to the repo convention: every other LLM call in this codebase
goes through ``LLMWrapper`` (LiteLLM + Instructor). This module talks to the raw
OpenAI **Responses API** (``client.responses.create`` / ``.parse``) on purpose —
the whole point of the exercise is to drive the reason→act→observe loop by hand so
each step is visible and captured in a trace. Do not "fix" this to use LLMWrapper.

Loop mechanics (stateful chaining):

1. ``responses.create`` with the transcript + the tool schemas. gpt-5 emits
   ``reasoning`` items and ``function_call`` items and then STOPS, waiting for us.
2. We read every ``function_call`` in ``response.output``, run the matching Python
   function, and send back one ``function_call_output`` per ``call_id``.
3. We re-call with ``previous_response_id`` and ONLY the new outputs — the server
   keeps the prior reasoning/function_call items and their ordering, which sidesteps
   the gpt-5 reasoning-item ordering pitfalls.
4. Repeat until a turn returns no ``function_call`` (natural stop) or we hit
   ``max_iterations`` (safeguard).
5. One final ``responses.parse`` turns the accumulated context into a validated
   ``AgentEstimate``.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from app.generation.agentic.agent_schemas import (
    AgentEstimate,
    AgentRunResult,
    AgentStep,
    AgentStructure,
    AgentTaskDerivation,
    AgentTaskHoursRun,
    AgentTaskRef,
    AgentTrace,
)
from app.generation.agentic.agent_tools import (
    HOURS_TOOL_SCHEMAS,
    TOOL_SCHEMAS,
    ConsensusFn,
    RecoveryRetrievalBackend,
    RetrievalBackend,
    default_retrieval_backend,
    dispatch_recovery_tool,
    dispatch_tool,
)
from app.foundation.prompts.loader import (
    render_agent_hours_recovery_prompt,
    render_agent_legacy_prompts,
    render_agent_structure_prompt,
)

log = structlog.get_logger()


def _extract_reasoning_summary(output: list[Any]) -> str | None:
    """Concatenate the reasoning-summary text emitted in one turn, if any.

    The Responses API surfaces a summary only when the call passes
    ``reasoning={"summary": "auto"}``; even then it may be empty for cheap efforts.
    """
    parts: list[str] = []
    for item in output:
        if getattr(item, "type", None) != "reasoning":
            continue
        for summary in getattr(item, "summary", None) or []:
            text = getattr(summary, "text", None)
            if text:
                parts.append(text)
    return " ".join(parts) if parts else None


def _function_calls(output: list[Any]) -> list[Any]:
    return [item for item in output if getattr(item, "type", None) == "function_call"]


def _is_summary_capability_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "summary" in message and any(
        marker in message
        for marker in ("unsupported", "not supported", "unknown parameter", "invalid")
    )


def _is_reasoning_capability_error(exc: Exception) -> bool:
    """True when the model rejects the Responses ``reasoning`` parameter entirely."""
    message = str(exc).lower()
    if "reasoning" not in message:
        return False
    return any(
        marker in message
        for marker in ("unsupported", "not supported", "unknown parameter", "invalid")
    )


async def _create_response(
    client: Any, *, summary_enabled: bool, **kwargs: Any
) -> tuple[Any, bool]:
    reasoning = {"effort": kwargs.pop("reasoning_effort")}
    if summary_enabled:
        reasoning["summary"] = "auto"
    try:
        response = await client.responses.create(reasoning=reasoning, **kwargs)
        return response, summary_enabled
    except Exception as exc:
        if not summary_enabled or not _is_summary_capability_error(exc):
            raise
        log.warning("agent_reasoning_summary_degraded")
        response = await client.responses.create(
            reasoning={"effort": reasoning["effort"]}, **kwargs
        )
        return response, False


async def run_estimation_agent(
    transcript: str,
    *,
    client: Any,
    model: str,
    reasoning_effort: str = "medium",
    max_iterations: int = 10,
    retrieval_backend: RetrievalBackend | None = None,
) -> AgentRunResult:
    """Run the manual agent loop over a transcript and return estimate + trace.

    ``client`` is an ``AsyncOpenAI`` instance (from ``get_async_openai_client()``).
    ``retrieval_backend`` overrides how ``search_budgets`` finds budgets — defaults
    to the real ``retrieve()`` pipeline; a stub can be injected for offline runs.
    """
    backend = retrieval_backend or default_retrieval_backend
    system_prompt, initial_user, final_user = render_agent_legacy_prompts(transcript)
    trace = AgentTrace()
    step_no = 0
    stopped_reason: str = "completed"

    log.info("agent_run_start", model=model, effort=reasoning_effort)
    response, summary_enabled = await _create_response(
        client,
        summary_enabled=True,
        reasoning_effort=reasoning_effort,
        model=model,
        instructions=system_prompt,
        input=[{"role": "user", "content": initial_user}],
        tools=TOOL_SCHEMAS,
        store=True,
    )
    iterations = 1

    while True:
        calls = _function_calls(response.output)
        if not calls:
            break
        if iterations >= max_iterations:
            stopped_reason = "max_iterations"
            log.warning("agent_max_iterations_reached", iterations=iterations)
            break

        # gpt-5 reasons ONCE per turn even when it emits several parallel tool
        # calls, so the summary belongs to the turn. Attach it to the first step
        # and mark the siblings as parallel calls of that same turn, rather than
        # repeating the whole reasoning block on each.
        reasoning_summary = _extract_reasoning_summary(response.output)
        first_step_in_turn = step_no + 1
        tool_outputs: list[dict[str, Any]] = []
        for call in calls:
            step_no += 1
            step_reasoning = (
                reasoning_summary
                if step_no == first_step_in_turn
                else f"(parallel tool call in the same turn as STEP {first_step_in_turn})"
            )
            name = getattr(call, "name", "unknown")
            try:
                raw_args = json.loads(call.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                raw_args = {}
                result: dict[str, Any] = {
                    "error": f"arguments were not valid JSON: {exc}"
                }
            else:
                try:
                    result = await dispatch_tool(name, raw_args, backend=backend)
                except Exception as exc:  # noqa: BLE001 — return the error so the model self-corrects.
                    log.warning("agent_tool_error", tool=name, error=str(exc)[:200])
                    result = {"error": f"{type(exc).__name__}: {exc}"}

            observation = (
                result.get("summary") or result.get("error") or json.dumps(result)[:200]
            )
            trace.steps.append(
                AgentStep(
                    step=step_no,
                    reasoning_summary=step_reasoning,
                    tool=name,
                    tool_args=raw_args,
                    observation=observation,
                )
            )
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                }
            )

        response, summary_enabled = await _create_response(
            client,
            summary_enabled=summary_enabled,
            reasoning_effort=reasoning_effort,
            model=model,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=TOOL_SCHEMAS,
            store=True,
        )
        iterations += 1

    estimate: AgentEstimate | None = None
    if stopped_reason != "max_iterations":
        try:
            parsed = await client.responses.parse(
                model=model,
                previous_response_id=response.id,
                input=[{"role": "user", "content": final_user}],
                text_format=AgentEstimate,
                store=True,
            )
            estimate = parsed.output_parsed
            iterations += 1
        except Exception as exc:  # noqa: BLE001 — a failed final parse is a stop reason, not a crash.
            log.error("agent_final_parse_failed", error=str(exc)[:300])
            stopped_reason = "no_final_estimate"

    if estimate is None and stopped_reason == "completed":
        stopped_reason = "no_final_estimate"

    log.info(
        "agent_run_done",
        iterations=iterations,
        steps=len(trace.steps),
        stopped_reason=stopped_reason,
        total_hours=(estimate.total_hours if estimate else None),
    )
    return AgentRunResult(
        estimate=estimate,
        trace=trace,
        iterations=iterations,
        stopped_reason=stopped_reason,  # type: ignore[arg-type]
    )


async def run_structure_agent(
    brief: object,
    *,
    client: Any,
    model: str,
    reasoning_effort: str,
    persona: str | None = None,
) -> tuple[AgentStructure, AgentTrace]:
    """Propose structure once, with structured output and no tools."""
    system_prompt, user_prompt = render_agent_structure_prompt(brief, persona)
    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": [{"role": "user", "content": user_prompt}],
        "text_format": AgentStructure,
        "reasoning": {"effort": reasoning_effort, "summary": "auto"},
        "store": True,
    }
    try:
        response = await client.responses.parse(**kwargs)
    except Exception as exc:
        if _is_summary_capability_error(exc):
            log.warning("agent_reasoning_summary_degraded")
            kwargs["reasoning"] = {"effort": reasoning_effort}
            try:
                response = await client.responses.parse(**kwargs)
            except Exception as nested:
                if not _is_reasoning_capability_error(nested):
                    raise
                log.warning("agent_reasoning_param_dropped", model=model)
                kwargs.pop("reasoning", None)
                response = await client.responses.parse(**kwargs)
        elif _is_reasoning_capability_error(exc):
            log.warning("agent_reasoning_param_dropped", model=model)
            kwargs.pop("reasoning", None)
            response = await client.responses.parse(**kwargs)
        else:
            raise
    structure = response.output_parsed
    if structure is None:
        raise RuntimeError("Structure response did not contain parsed output.")
    task_count = sum(len(module.tasks) for module in structure.modules)
    trace = AgentTrace(
        steps=[
            AgentStep(
                step=1,
                reasoning_summary=_extract_reasoning_summary(response.output),
                tool="propose_structure",
                tool_args={},
                observation=(
                    f"proposed {len(structure.modules)} modules and {task_count} tasks"
                ),
            )
        ]
    )
    return structure, trace


async def run_task_hours_recovery_agent(
    flagged_tasks: list[AgentTaskRef],
    *,
    client: Any,
    model: str,
    reasoning_effort: str,
    max_iterations: int,
    backend: RecoveryRetrievalBackend,
    consensus: ConsensusFn,
    persona: str | None = None,
) -> AgentTaskHoursRun:
    """Recover flagged task hours through tools, never terminal free-form numbers."""
    if not flagged_tasks:
        return AgentTaskHoursRun(iterations=0)

    system_prompt, user_prompt = render_agent_hours_recovery_prompt(
        flagged_tasks, persona
    )
    trace = AgentTrace()
    derivations: list[AgentTaskDerivation] = []
    response, summary_enabled = await _create_response(
        client,
        summary_enabled=True,
        reasoning_effort=reasoning_effort,
        model=model,
        instructions=system_prompt,
        input=[{"role": "user", "content": user_prompt}],
        tools=HOURS_TOOL_SCHEMAS,
        store=True,
    )
    iterations = 1
    step_no = 0

    while True:
        calls = _function_calls(response.output)
        if not calls:
            return AgentTaskHoursRun(
                derivations=derivations,
                trace=trace,
                iterations=iterations,
                stopped_reason="completed",
            )
        if iterations >= max_iterations:
            return AgentTaskHoursRun(
                derivations=derivations,
                trace=trace,
                iterations=iterations,
                stopped_reason="max_iterations",
            )

        reasoning_summary = _extract_reasoning_summary(response.output)
        first_step = step_no + 1
        outputs: list[dict[str, Any]] = []
        for call in calls:
            step_no += 1
            name = getattr(call, "name", "unknown")
            try:
                raw_args = json.loads(call.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                raw_args = {}
                result = {"error": f"arguments were not valid JSON: {exc}"}
            else:
                try:
                    result = await dispatch_recovery_tool(
                        name,
                        raw_args,
                        backend=backend,
                        consensus=consensus,
                    )
                    if name == "derive_task_hours":
                        derivations.append(AgentTaskDerivation.model_validate(result))
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "agent_recovery_tool_error",
                        tool=name,
                        error_type=type(exc).__name__,
                    )
                    result = {"error": f"{type(exc).__name__}: {exc}"}
            trace.steps.append(
                AgentStep(
                    step=step_no,
                    reasoning_summary=reasoning_summary
                    if step_no == first_step
                    else f"(parallel tool call in the same turn as STEP {first_step})",
                    tool=name,
                    tool_args=raw_args,
                    observation=str(
                        result.get("summary")
                        or result.get("error")
                        or f"{name} completed"
                    )[:200],
                )
            )
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                }
            )

        response, summary_enabled = await _create_response(
            client,
            summary_enabled=summary_enabled,
            reasoning_effort=reasoning_effort,
            model=model,
            previous_response_id=response.id,
            input=outputs,
            tools=HOURS_TOOL_SCHEMAS,
            store=True,
        )
        iterations += 1
