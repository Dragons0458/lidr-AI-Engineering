#!/usr/bin/env python3
"""Session 12 — run the hand-written estimation agent over a transcript.

This is the deliverable generator: it feeds a meeting transcript to the manual
agent loop (``app/generation/agentic/agent_loop.py``), prints the step-by-step
trace (reasoning → action → observation) and the final structured estimate, and
optionally writes them to a file.

Cost discipline (from the statement): debug the LOOP MECHANICS cheaply first with
``gpt-5-mini`` and the simple transcript, then switch to ``gpt-5`` / ``medium`` for
the real run on the complex transcript.

    # 1) cheap loop debugging (real retrieval, needs the stack up + task corpus)
    docker compose exec api python scripts/run_agent_s12.py \\
        exercises/session-12/sample_transcript_simple.txt --model gpt-5-mini --effort minimal

    # 2) offline loop debugging with the student stub (NO database needed)
    uv run python scripts/run_agent_s12.py \\
        exercises/session-12/sample_transcript_simple.txt --model gpt-5-mini --stub

    # 3) the real deliverable run
    docker compose exec api python scripts/run_agent_s12.py \\
        exercises/session-12/sample_transcript_complex.txt --model gpt-5 --effort medium \\
        --out exercises/session-12/example_trace_complex.txt

``search_budgets`` wraps the real S9/S10 ``retrieve()`` pipeline by default, so the
real runs need the stack up and the historical-task corpus ingested
(``scripts/build_task_corpus.py --ingest``). ``--stub`` swaps in the offline
reference retrieval so the loop can be exercised without a database.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings  # noqa: E402
from app.dependencies import get_async_openai_client  # noqa: E402
from app.domain.agent_estimation import (  # noqa: E402
    agent_estimate_task_hours,
    agent_propose_structure,
)
from app.generation.agentic.agent_loop import (  # noqa: E402
    run_estimation_agent,
    run_structure_agent,
    run_task_hours_recovery_agent,
)
from app.generation.agentic.agent_schemas import (  # noqa: E402
    AgentRunResult,
    AgentTaskRef,
    SearchBudgetsArgs,
)
from app.generation.rag.agent_retrieval import make_retrieval_backend  # noqa: E402
from app.generation.rag.prompt_builder import build_structure_user_message  # noqa: E402
from app.generation.rag.query_reformulator import reformulate_query  # noqa: E402
from app.generation.rag.schemas import (  # noqa: E402
    TaskHoursModuleInput,
    TaskHoursTaskInput,
)
from app.generation.rag.task_hours import distance_weighted_consensus  # noqa: E402

STUB_PATH = REPO_ROOT / "exercises" / "session-12" / "reference_retrieval.py"


def _load_stub_backend():
    """Load the student safety-net retrieval stub and adapt it to a RetrievalBackend."""
    spec = importlib.util.spec_from_file_location("s12_reference_retrieval", STUB_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load stub retrieval from {STUB_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def stub_backend(args: SearchBudgetsArgs) -> list[dict]:
        filters = args.filters.model_dump() if args.filters else None
        return module.search_budgets_stub(args.query, filters)

    return stub_backend


def _load_recovery_stub_backend():
    """Load the same stub behind the recovery query/sectors signature."""
    legacy = _load_stub_backend()

    async def recovery(query: str, sectors: list[str] | None) -> list[dict]:
        raw = {
            "query": query,
            "filters": {"sectors": sectors, "component_type": None},
        }
        return await legacy(SearchBudgetsArgs.model_validate(raw))

    return recovery


def _render(result: AgentRunResult) -> str:
    lines = [
        "=" * 78,
        "AGENT TRACE",
        "=" * 78,
        result.trace.render(),
        "",
        "=" * 78,
        f"FINAL ESTIMATE  (iterations={result.iterations}, stopped={result.stopped_reason})",
        "=" * 78,
    ]
    estimate = result.estimate
    if estimate is None:
        lines.append("(the agent stopped without producing a structured estimate)")
        return "\n".join(lines)

    for component in estimate.components:
        cited = ", ".join(str(c) for c in component.cited_chunk_ids) or "none"
        lines.append(
            f"  - {component.name}: {component.estimated_hours}h  [sources: {cited}]"
        )
        lines.append(f"      {component.rationale}")
    lines.append("")
    lines.append(
        f"  TOTAL: {estimate.total_hours}h    confidence: {estimate.confidence}"
    )
    if estimate.assumptions:
        lines.append("  assumptions:")
        for assumption in estimate.assumptions:
            lines.append(f"    · {assumption}")
    return "\n".join(lines)


async def _main_async(args: argparse.Namespace) -> int:
    transcript_path = Path(args.transcript)
    if not transcript_path.is_file():
        print(f"ERROR: transcript not found: {transcript_path}", file=sys.stderr)
        return 1

    client = get_async_openai_client()
    if client is None:
        print(
            "ERROR: OPENAI_API_KEY is not set — the agent needs the OpenAI Responses API.",
            file=sys.stderr,
        )
        return 1

    transcript = transcript_path.read_text(encoding="utf-8")

    print(f"transcript : {transcript_path}")
    print(
        f"model      : {args.model}   effort: {args.effort}   backend: "
        f"{'stub' if args.stub else 'retrieve() pipeline'}"
    )
    print()

    if args.workflow == "legacy":
        result = await run_estimation_agent(
            transcript,
            client=client,
            model=args.model,
            reasoning_effort=args.effort,
            max_iterations=args.max_iterations,
            retrieval_backend=_load_stub_backend() if args.stub else None,
        )
        rendered = _render(result)
    elif args.workflow == "hybrid":
        query = await reformulate_query(transcript)
        structure_result = await agent_propose_structure(
            query,
            client=client,
            model=args.model,
            reasoning_effort=args.effort,
            persona=args.persona,
        )
        modules = [
            TaskHoursModuleInput(
                name=module.name,
                tasks=[
                    TaskHoursTaskInput(
                        name=task.name,
                        description=task.description,
                    )
                    for task in module.tasks
                ],
            )
            for module in structure_result.estimate.modules
        ]
        settings = get_settings()
        hours_result = await agent_estimate_task_hours(
            modules,
            client=client,
            model=args.model,
            reasoning_effort=args.effort,
            max_iterations=args.max_iterations,
            top_k=settings.TASK_HOURS_TOP_K,
            distance_threshold=settings.TASK_HOURS_DISTANCE_THRESHOLD,
            search_mode=settings.RETRIEVAL_SEARCH_MODE,
            rerank=settings.RERANKER_ENABLED,
            persona=args.persona,
            recovery_reliability_threshold=(
                settings.AGENT_RECOVERY_RELIABILITY_THRESHOLD
            ),
        )
        rendered = (
            "STRUCTURE TRACE\n"
            f"{structure_result.agent_trace.render() if structure_result.agent_trace else '(none)'}\n\n"
            f"STRUCTURE\n{structure_result.estimate.model_dump_json(indent=2)}\n\n"
            "HOURS TRACE\n"
            f"{hours_result.agent_trace.render() if hours_result.agent_trace else '(none)'}\n\n"
            f"HOURS\n{hours_result.model_dump_json(indent=2)}"
        )
    else:
        query = await reformulate_query(transcript)
        structure, structure_trace = await run_structure_agent(
            build_structure_user_message(query),
            client=client,
            model=args.model,
            reasoning_effort=args.effort,
            persona=args.persona,
        )
        flagged = [
            AgentTaskRef(
                task_ref=f"task-{index}",
                module=module.name,
                task=task.name,
                description=task.description,
                reason="recovery demo",
            )
            for index, (module, task) in enumerate(
                (module, task) for module in structure.modules for task in module.tasks
            )
        ]
        settings = get_settings()
        backend = (
            _load_recovery_stub_backend()
            if args.stub
            else make_retrieval_backend(
                top_k=settings.AGENT_SEARCH_TOP_K,
                distance_threshold=settings.AGENT_SEARCH_DISTANCE_THRESHOLD,
                search_mode=settings.RETRIEVAL_SEARCH_MODE,
                rerank=settings.RERANKER_ENABLED,
            )
        )
        recovery = await run_task_hours_recovery_agent(
            flagged,
            client=client,
            model=args.model,
            reasoning_effort=args.effort,
            max_iterations=args.max_iterations,
            backend=backend,
            consensus=distance_weighted_consensus,
            persona=args.persona,
        )
        rendered = (
            f"STRUCTURE TRACE\n{structure_trace.render()}\n\n"
            f"STRUCTURE\n{structure.model_dump_json(indent=2)}\n\n"
            f"RECOVERY TRACE\n{recovery.trace.render()}\n\n"
            f"DERIVATIONS\n{recovery.model_dump_json(indent=2)}"
        )
    print(rendered)

    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
        print(f"\n(trace written to {args.out})")
    return 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the Session 12 estimation agent.")
    parser.add_argument("transcript", help="Path to a meeting transcript .txt file.")
    parser.add_argument(
        "--workflow",
        choices=["hybrid", "recovery-demo", "legacy"],
        default="hybrid",
        help="Workflow to run (default: hybrid).",
    )
    parser.add_argument(
        "--model",
        default=settings.AGENT_MODEL,
        help=f"OpenAI model (default {settings.AGENT_MODEL}).",
    )
    parser.add_argument(
        "--effort",
        default=settings.AGENT_REASONING_EFFORT,
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort for the Responses API.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=settings.AGENT_MAX_ITERATIONS,
        help="Loop safeguard: max Responses API round-trips.",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use the offline reference retrieval stub (no database).",
    )
    parser.add_argument(
        "--out", help="Write the rendered trace + estimate to this file."
    )
    parser.add_argument("--persona", help="Optional agent working persona.")
    args = parser.parse_args()
    if args.stub and args.workflow == "hybrid":
        parser.error("--stub is not supported with --workflow hybrid")
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
