#!/usr/bin/env python3
"""Session 14 supervisor runner and reproducible evidence generator.

Offline evidence (no network, no model calls)::

    uv run python scripts/run_agent_s14.py --generate-evidence

One offline run with an in-memory checkpointer::

    uv run python scripts/run_agent_s14.py \
        exercises/session-14/sample_transcript_happy_path.txt \
        --memory --stub

Live HTTP start -> optional review resume -> final state::

    uv run python scripts/run_agent_s14.py \
        exercises/session-14/sample_transcript_edge_case.txt \
        --base-url http://localhost:8000 \
        --api-key "$ESTIMATE_API_KEY" \
        --decision approve
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command  # noqa: E402

from app.generation.agentic.agent_schemas import (  # noqa: E402
    AgentModuleNode,
    AgentStructure,
    AgentTaskNode,
)
from app.generation.agentic.graph.supervisor_build import AGENT_NODE_NAMES  # noqa: E402
from app.generation.agentic.graph.supervisor_nodes import (  # noqa: E402
    SupervisorDecision,
    SupervisorDeps,
    make_supervisor_nodes,
)
from app.generation.agentic.graph.supervisor_privilege import (  # noqa: E402
    AGENT_PRIVILEGES,
    guarded_dispatch,
)
from app.generation.agentic.graph.supervisor_state import (  # noqa: E402
    SupervisorState,
    detect_review_risks,
)
from app.generation.rag.schemas import EstimationQuery  # noqa: E402

SUPERVISOR_PREFIX = "/v1/estimate/agent/supervisor"
EXERCISE_DIR = REPO_ROOT / "exercises" / "session-14"
DEFAULT_TRANSCRIPT = EXERCISE_DIR / "sample_transcript_edge_case.txt"
STUB_PATH = REPO_ROOT / "exercises" / "session-12" / "reference_retrieval.py"


@dataclass(frozen=True)
class RunEvidence:
    """Final state plus the interrupt payload that existed before resume."""

    state: dict[str, Any]
    review_triggered: bool = False
    review_reasons: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    decision: str | None = None


@dataclass(frozen=True)
class EvidenceCase:
    name: str
    transcript: Path
    violate: bool = False
    decision: str = "approve"


EVIDENCE_CASES = (
    EvidenceCase("happy", EXERCISE_DIR / "sample_transcript_happy_path.txt"),
    EvidenceCase("edge_case", EXERCISE_DIR / "sample_transcript_edge_case.txt"),
    EvidenceCase(
        "violate",
        EXERCISE_DIR / "sample_transcript_happy_path.txt",
        violate=True,
    ),
)


def build_payload(transcript: str, estimation_id: str | None) -> dict[str, str]:
    """Build the stable public start payload used by the HTTP mode."""
    return {
        "estimation_id": estimation_id or f"s14-{uuid4()}",
        "transcript": transcript,
    }


def _load_stub_backend():
    """Load the standalone S12 keyword stub without importing it as app code."""
    if not STUB_PATH.exists():

        async def empty(_args):
            return []

        return empty

    spec = importlib.util.spec_from_file_location("reference_retrieval", STUB_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load retrieval stub: {STUB_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def backend(args):
        query = args.query if hasattr(args, "query") else str(args)
        sectors = None
        if hasattr(args, "filters") and args.filters is not None:
            sectors = getattr(args.filters, "sectors", None)
        return module.search_budgets_stub(query, {"sectors": sectors})

    return backend


def _offline_deps(*, force_review: bool) -> SupervisorDeps:
    """Deterministic collaborators: no database, model, Redis, or network."""

    async def reformulate(_transcript: str) -> EstimationQuery:
        return EstimationQuery(
            function="Software estimation evidence",
            technologies=["Python"],
            sector=None if force_review else "logistics",
        )

    async def propose_structure(_brief: EstimationQuery) -> AgentStructure:
        return AgentStructure(
            modules=[
                AgentModuleNode(
                    name="Core",
                    tasks=[
                        AgentTaskNode(name="Auth", description="login"),
                        AgentTaskNode(name="API", description="REST backend"),
                    ],
                )
            ],
            confidence="medium" if force_review else "high",
            reasoning="offline evidence structure",
        )

    async def route(_digest: str) -> SupervisorDecision:
        raise RuntimeError("offline evidence router uses deterministic fallback")

    async def empty_backend(_args):
        return []

    return SupervisorDeps(
        reformulate=reformulate,
        propose_structure=propose_structure,
        retrieval_backend=empty_backend if force_review else _load_stub_backend(),
        route_with_model=route,
        confidence_threshold=0.6,
        min_grounded_ratio=0.5,
        out_of_range_factor=2.0,
        max_steps=8,
        privilege_strict=False,
        grounding_max_distance=0.55,
    )


def _compile(nodes: dict[str, Any], checkpointer: Any):
    builder = StateGraph(SupervisorState)
    builder.add_node(
        "supervisor",
        nodes["supervisor"],
        destinations=(*AGENT_NODE_NAMES, "human_review_gate"),
    )
    for name in AGENT_NODE_NAMES:
        builder.add_node(name, nodes[name])
    builder.add_node("human_review_gate", nodes["human_review_gate"])
    builder.add_edge(START, "supervisor")
    for name in AGENT_NODE_NAMES:
        builder.add_edge(name, "supervisor")
    builder.add_edge("human_review_gate", END)
    return builder.compile(checkpointer=checkpointer)


def _install_privilege_probe(nodes: dict[str, Any], deps: SupervisorDeps) -> None:
    """Make the searcher attempt one forbidden tool before its legitimate work."""
    original = nodes["budget_searcher"]

    async def probing(state):
        _result, denial = await guarded_dispatch(
            "budget_searcher",
            "validate_estimate",
            {"components": [], "total_hours": 0},
            step=int(state.get("supervisor_steps") or 0),
            estimation_id=state.get("estimation_id"),
            privilege_strict=deps.privilege_strict,
        )
        update = await original(state)
        return {
            **update,
            "agent_contributions": [
                denial,
                *(update.get("agent_contributions") or []),
            ],
        }

    nodes["budget_searcher"] = probing


async def run_memory_flow(
    transcript: str,
    *,
    estimation_id: str,
    decision: str,
    violate: bool = False,
) -> RunEvidence:
    """Run the complete graph locally and capture any interrupt before resuming."""
    risk_flags = detect_review_risks({"transcript": transcript})
    deps = _offline_deps(force_review=bool(risk_flags))
    nodes = make_supervisor_nodes(deps)
    if violate:
        _install_privilege_probe(nodes, deps)

    graph = _compile(nodes, MemorySaver())
    config = {"configurable": {"thread_id": f"s14:{estimation_id}"}}
    await graph.ainvoke(
        {"transcript": transcript, "estimation_id": estimation_id},
        config,
    )

    review_triggered = False
    review_reasons: list[str] = []
    captured_risks: list[str] = []
    answered: str | None = None
    for _ in range(4):
        snapshot = await graph.aget_state(config)
        if not snapshot.next:
            return RunEvidence(
                state=dict(snapshot.values),
                review_triggered=review_triggered,
                review_reasons=review_reasons,
                risk_flags=captured_risks,
                decision=answered,
            )
        interrupts = getattr(snapshot, "interrupts", None) or ()
        if not interrupts:
            await graph.ainvoke(None, config)
            continue
        payload = interrupts[0].value or {}
        review_triggered = True
        review_reasons = list(payload.get("reasons") or [])
        captured_risks = list(payload.get("risk_flags") or [])
        answered = decision
        await graph.ainvoke(
            Command(
                resume={
                    "decision": decision,
                    "note": "auto-decided by run_agent_s14.py",
                }
            ),
            config,
        )
    raise RuntimeError("Supervisor did not complete after four resume attempts")


def run_http_flow(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, str],
    decision: str,
) -> RunEvidence:
    """Start through FastAPI, resume a pending review, then read the checkpoint."""
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    root = base_url.rstrip("/")
    estimation_id = payload["estimation_id"]

    response = client.post(
        f"{root}{SUPERVISOR_PREFIX}",
        headers=headers,
        json=payload,
    )
    response.raise_for_status()
    state = response.json()

    review_triggered = False
    review_reasons: list[str] = []
    risk_flags: list[str] = []
    answered: str | None = None
    for _ in range(4):
        if state.get("state") != "paused":
            break
        pending = state.get("pending_review") or {}
        if not pending:
            raise RuntimeError("Paused supervisor response has no pending_review")
        review_triggered = True
        review_reasons = list(pending.get("reasons") or [])
        risk_flags = list(pending.get("risk_flags") or [])
        answered = decision
        resume = client.post(
            f"{root}{SUPERVISOR_PREFIX}/{estimation_id}/resume",
            headers=headers,
            json={
                "decision": decision,
                "note": "auto-decided by run_agent_s14.py",
            },
        )
        resume.raise_for_status()
        state = resume.json()
    else:
        raise RuntimeError("Supervisor did not complete after four resume attempts")

    final = client.get(
        f"{root}{SUPERVISOR_PREFIX}/{estimation_id}/state",
        headers=headers,
    )
    final.raise_for_status()
    return RunEvidence(
        state=final.json(),
        review_triggered=review_triggered,
        review_reasons=review_reasons,
        risk_flags=risk_flags,
        decision=answered,
    )


def render_evidence(evidence: RunEvidence) -> str:
    """Render a stable, human-readable audit artifact."""
    state = evidence.state
    lines = [
        "=" * 78,
        "SESSION 14 — SUPERVISOR MULTI-AGENT RUN",
        "=" * 78,
        "",
        f"estimation_id = {state.get('estimation_id', 'n/a')}",
        "",
        "ROUTING (supervisor decisions)",
        "-" * 78,
    ]
    for row in state.get("routing_history") or []:
        lines.append(
            f"  {row['step'] + 1}. supervisor → {row['next_agent']:<24} "
            f"[{row.get('source', '?')}]  {row.get('reason', '')[:90]}"
        )
    lines += ["", "TOOL PRIVILEGE (declared allowlists)", "-" * 78]
    for agent, tools in AGENT_PRIVILEGES.items():
        rendered = ", ".join(sorted(tools)) if tools else "(no tools)"
        lines.append(f"  {agent:<24} : {rendered}")

    lines += ["", "AUDIT TRAIL (agent_contributions)", "-" * 78]
    for row in state.get("agent_contributions") or []:
        marker = {"ok": "ok", "denied": "DENIED", "error": "ERROR"}.get(
            row.get("outcome", "?"), "?"
        )
        lines.append(
            f"  [{marker:^6}] {row.get('agent', '?'):<24} "
            f"{row.get('action', '?'):<28} {row.get('summary', '')[:60]}"
        )

    lines += ["", "HUMAN REVIEW", "-" * 78]
    if evidence.review_triggered:
        lines.append("  triggered: YES")
        for reason in evidence.review_reasons:
            lines.append(f"    - {reason}")
        if evidence.risk_flags:
            lines.append("  risk flags:")
            for flag in evidence.risk_flags:
                lines.append(f"    - {flag}")
        lines.append(f"  decision : {evidence.decision or '(not resumed)'}")
    else:
        confidence = state.get("confidence")
        confidence_text = f"{confidence:.2f}" if confidence is not None else "n/a"
        lines.append(
            f"  triggered: no (confidence {confidence_text}, all conditions clear)"
        )

    lines += ["", "ESTIMATE", "-" * 78]
    estimate = state.get("estimate") or {}
    for component in estimate.get("components") or []:
        hours = component.get("estimated_hours")
        lines.append(f"  {component.get('name', '?'):<40} {hours}h")
    lines.append(f"  {'TOTAL':<40} {estimate.get('total_hours')}h")
    lines.append(f"  status = {state.get('status')}")
    return "\n".join(lines)


def write_evidence(path: Path, evidence: RunEvidence) -> None:
    """Write text or JSON evidence with identical semantic content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        payload = {
            "review": {
                "triggered": evidence.review_triggered,
                "reasons": evidence.review_reasons,
                "risk_flags": evidence.risk_flags,
                "decision": evidence.decision,
            },
            "state": evidence.state,
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return
    path.write_text(render_evidence(evidence) + "\n", encoding="utf-8")


async def generate_evidence(output_dir: Path = EXERCISE_DIR) -> list[Path]:
    """Regenerate the committed/local S14 evidence kit deterministically."""
    written: list[Path] = []
    for case in EVIDENCE_CASES:
        transcript = case.transcript.read_text(encoding="utf-8")
        evidence = await run_memory_flow(
            transcript,
            estimation_id=f"s14-evidence-{case.name}",
            decision=case.decision,
            violate=case.violate,
        )
        path = output_dir / f"example_run_{case.name}.txt"
        write_evidence(path, evidence)
        written.append(path)
    return written


def _resolve_transcript(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> Path:
    if args.transcript_positional and args.transcript_option:
        parser.error("Use either the positional transcript or --transcript, not both")
    path = args.transcript_option or args.transcript_positional or DEFAULT_TRANSCRIPT
    if not path.is_file():
        parser.error(f"Transcript not found: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript_positional", nargs="?", type=Path)
    parser.add_argument("--transcript", dest="transcript_option", type=Path)
    parser.add_argument("--memory", action="store_true", help="Use MemorySaver locally")
    parser.add_argument(
        "--stub", action="store_true", help="Use deterministic offline deps"
    )
    parser.add_argument("--violate", action="store_true", help="Demonstrate a denial")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--estimation-id", default=None)
    parser.add_argument(
        "--decision",
        choices=["approve", "adjust", "reject"],
        default="approve",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--generate-evidence",
        action="store_true",
        help="Regenerate happy, edge_case and violate artifacts offline",
    )
    parser.add_argument("--evidence-dir", type=Path, default=EXERCISE_DIR)
    args = parser.parse_args(argv)

    if args.generate_evidence:
        paths = asyncio.run(generate_evidence(args.evidence_dir))
        for path in paths:
            print(f"wrote {path}")
        return 0

    transcript_path = _resolve_transcript(args, parser)
    transcript = transcript_path.read_text(encoding="utf-8")
    estimation_id = args.estimation_id or f"s14-{uuid4()}"

    if args.memory:
        if not args.stub:
            print("--memory implies deterministic --stub dependencies", file=sys.stderr)
        evidence = asyncio.run(
            run_memory_flow(
                transcript,
                estimation_id=estimation_id,
                decision=args.decision,
                violate=args.violate,
            )
        )
    else:
        if args.stub:
            parser.error("--stub requires --memory")
        if args.violate:
            parser.error("--violate is only available with --memory --stub")
        api_key = args.api_key or os.environ.get("ESTIMATE_API_KEY")
        if not api_key:
            parser.error("Provide --api-key or set ESTIMATE_API_KEY")
        payload = build_payload(transcript, estimation_id)
        try:
            with httpx.Client(timeout=args.timeout) as client:
                evidence = run_http_flow(
                    client,
                    base_url=args.base_url or "http://localhost:8000",
                    api_key=api_key,
                    payload=payload,
                    decision=args.decision,
                )
        except httpx.HTTPStatusError as exc:
            print(f"HTTP {exc.response.status_code}", file=sys.stderr)
            print(exc.response.text, file=sys.stderr)
            return 1

    rendered = render_evidence(evidence)
    print(rendered)
    if args.out:
        write_evidence(args.out, evidence)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
