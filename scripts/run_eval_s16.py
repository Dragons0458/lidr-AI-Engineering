#!/usr/bin/env python3
"""Session 16 production eval harness.

Calls the *deployed* AI service (real model, real tokens) and grades the
response against ``evals/golden_set_s16.json``. This is evaluation, not a
unit test — keep it out of CI.

    uv run python scripts/run_eval_s16.py --dry-run
    uv run python scripts/run_eval_s16.py --arm rag --case S16-01
    uv run python scripts/run_eval_s16.py --arm both --label baseline-s16

Never sends ``idempotency_key``. 429s are retried and do not count as system
errors. Default pacing is 7s because both estimate routes are 10/minute.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.production.adapters import (  # noqa: E402
    normalize_graph_state,
    normalize_rag_response,
)
from evals.production.grading import evaluate_case  # noqa: E402
from evals.production.metrics import ab_compare, aggregate  # noqa: E402
from evals.production.reporting import render_markdown  # noqa: E402
from evals.production.schemas import (  # noqa: E402
    ArmReport,
    CaseEvaluation,
    EvalReport,
    GoldenCase,
    GoldenSet,
    Outcome,
)
from evals.production.validate import load_golden_set, validate_golden_set  # noqa: E402

GRAPH_PREFIX = "/v1/estimate/agent/graph"
RAG_PATH = "/v1/estimate/from-transcript"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _headers(api_key: str, request_id: str) -> dict[str, str]:
    return {
        "X-API-Key": api_key,
        "X-Request-ID": request_id,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def canned_decision(gate: str) -> dict:
    if gate == "structure_review":
        return {"approved": True}
    if gate == "final_review":
        return {"validated": True, "want_proposal": False}
    raise ValueError(f"Unknown gate: {gate!r}")


def request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict | None = None,
    max_retries: int = 4,
) -> httpx.Response:
    delay = 5.0
    response: httpx.Response | None = None
    for _attempt in range(max_retries + 1):
        response = client.request(method, url, headers=headers, json=json_body)
        if response.status_code != 429:
            return response
        retry_after = response.headers.get("Retry-After")
        try:
            wait = float(retry_after) if retry_after else delay
        except ValueError:
            wait = delay
        time.sleep(max(wait, 1.0))
        delay = min(delay * 2, 60.0)
    assert response is not None
    return response


def _join_cost(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    request_id_prefix: str,
) -> float | None:
    url = urljoin(base_url.rstrip("/") + "/", "v1/observability/requests")
    try:
        response = client.get(
            url,
            headers=_headers(api_key, f"{request_id_prefix}-cost"),
            params={"request_id_prefix": request_id_prefix, "limit": 50},
        )
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    rows = response.json()
    if not isinstance(rows, list):
        return None
    total = 0.0
    seen = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get("estimated_cost_usd")
        if value is None:
            continue
        seen = True
        total += float(value)
    return total if seen else None


def fetch_environment(
    client: httpx.Client, *, base_url: str, api_key: str
) -> dict[str, Any]:
    env: dict[str, Any] = {"base_url": base_url}
    root = base_url.rstrip("/")
    headers = _headers(api_key, f"s16-env-{uuid.uuid4().hex[:8]}")
    try:
        models = client.get(f"{root}/api/v1/config/models", headers=headers)
        if models.status_code == 200:
            payload = models.json()
            gen = (payload.get("models") or {}).get("GENERATION_MODEL") or {}
            env["generation_model"] = gen.get("effective")
            env["app_env"] = payload.get("app_env")
            env["models"] = {
                key: (value or {}).get("effective")
                for key, value in (payload.get("models") or {}).items()
                if isinstance(value, dict)
            }
    except httpx.HTTPError:
        env["generation_model"] = None
    try:
        retrieval = client.get(f"{root}/api/v1/config/retrieval", headers=headers)
        if retrieval.status_code == 200:
            env["retrieval_config"] = retrieval.json()
    except httpx.HTTPError:
        env["retrieval_config"] = None
    return env


def run_rag_case(
    client: httpx.Client,
    case: GoldenCase,
    transcript: str,
    *,
    base_url: str,
    api_key: str,
    run_id: str,
) -> Outcome:
    request_id = f"s16-{run_id}-{case.id}-1"
    url = urljoin(base_url.rstrip("/") + "/", RAG_PATH.lstrip("/"))
    started = time.perf_counter()
    try:
        response = request_with_retry(
            client,
            "POST",
            url,
            headers=_headers(api_key, request_id),
            json_body={"transcript": transcript},
        )
    except httpx.HTTPError as exc:
        return normalize_rag_response(
            None,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )
    latency_ms = (time.perf_counter() - started) * 1000
    if response.status_code == 429:
        return normalize_rag_response(
            None,
            latency_ms=latency_ms,
            http_status=429,
            throttled=True,
        )
    if response.status_code >= 400:
        return normalize_rag_response(
            None,
            latency_ms=latency_ms,
            http_status=response.status_code,
            error=f"HTTP {response.status_code}",
        )
    outcome = normalize_rag_response(
        response.json(),
        latency_ms=latency_ms,
        http_status=response.status_code,
    )
    outcome.cost_usd = _join_cost(
        client,
        base_url=base_url,
        api_key=api_key,
        request_id_prefix=f"s16-{run_id}-{case.id}",
    )
    return outcome


def run_graph_case(
    client: httpx.Client,
    case: GoldenCase,
    transcript: str,
    *,
    base_url: str,
    api_key: str,
    run_id: str,
) -> Outcome:
    estimation_id = f"s16-{run_id}-{case.id}"
    root = base_url.rstrip("/")
    seq = 1
    llm_calls = 0
    started = time.perf_counter()

    def _rid() -> str:
        nonlocal seq
        header = f"s16-{run_id}-{case.id}-{seq}"
        seq += 1
        return header

    try:
        start = request_with_retry(
            client,
            "POST",
            f"{root}{GRAPH_PREFIX}",
            headers=_headers(api_key, _rid()),
            json_body={"estimation_id": estimation_id, "transcript": transcript},
        )
    except httpx.HTTPError as exc:
        return normalize_graph_state(
            None,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
            llm_calls=llm_calls,
        )
    llm_calls += 1
    if start.status_code == 503:
        return normalize_graph_state(
            None,
            latency_ms=(time.perf_counter() - started) * 1000,
            http_status=503,
            skipped=True,
            skip_reason="runtime_unavailable",
            error="HTTP 503",
            llm_calls=llm_calls,
        )
    if start.status_code == 429:
        return normalize_graph_state(
            None,
            latency_ms=(time.perf_counter() - started) * 1000,
            http_status=429,
            throttled=True,
            llm_calls=llm_calls,
        )
    if start.status_code >= 400:
        return normalize_graph_state(
            None,
            latency_ms=(time.perf_counter() - started) * 1000,
            http_status=start.status_code,
            error=f"HTTP {start.status_code}",
            llm_calls=llm_calls,
        )

    state = start.json()
    while state.get("state") == "paused":
        pending = state.get("pending_gate") or {}
        gate = pending.get("gate", "unknown")
        resume = request_with_retry(
            client,
            "POST",
            f"{root}{GRAPH_PREFIX}/{estimation_id}/resume",
            headers=_headers(api_key, _rid()),
            json_body={"decision": canned_decision(gate)},
        )
        llm_calls += 1
        if resume.status_code == 429:
            return normalize_graph_state(
                None,
                latency_ms=(time.perf_counter() - started) * 1000,
                http_status=429,
                throttled=True,
                llm_calls=llm_calls,
            )
        if resume.status_code == 503:
            return normalize_graph_state(
                None,
                latency_ms=(time.perf_counter() - started) * 1000,
                http_status=503,
                skipped=True,
                skip_reason="runtime_unavailable",
                error="HTTP 503",
                llm_calls=llm_calls,
            )
        if resume.status_code >= 400:
            return normalize_graph_state(
                None,
                latency_ms=(time.perf_counter() - started) * 1000,
                http_status=resume.status_code,
                error=f"HTTP {resume.status_code}",
                llm_calls=llm_calls,
            )
        state = resume.json()

    snapshot = request_with_retry(
        client,
        "GET",
        f"{root}{GRAPH_PREFIX}/{estimation_id}/state",
        headers=_headers(api_key, _rid()),
    )
    llm_calls += 1
    latency_ms = (time.perf_counter() - started) * 1000
    if snapshot.status_code >= 400:
        return normalize_graph_state(
            state if isinstance(state, dict) else None,
            latency_ms=latency_ms,
            http_status=snapshot.status_code,
            error=f"HTTP {snapshot.status_code}",
            llm_calls=llm_calls,
        )
    outcome = normalize_graph_state(
        snapshot.json(),
        latency_ms=latency_ms,
        http_status=snapshot.status_code,
        llm_calls=llm_calls,
    )
    outcome.cost_usd = _join_cost(
        client,
        base_url=base_url,
        api_key=api_key,
        request_id_prefix=f"s16-{run_id}-{case.id}",
    )
    return outcome


def _print_case(case: GoldenCase, evaluation: CaseEvaluation) -> None:
    predicted = (
        "—"
        if evaluation.predicted_engineer_days is None
        else evaluation.predicted_engineer_days
    )
    signal = evaluation.abstention_signal or "—"
    print(
        f"  {case.id} [{evaluation.arm}] expected={case.expected_engineer_days} "
        f"range={case.acceptable_range} predicted={predicted} "
        f"abstained={evaluation.abstained} ({signal}) "
        f"→ {evaluation.verdict.upper()}  ({evaluation.latency_ms:.0f} ms)"
    )


def run_arm(
    client: httpx.Client,
    golden: GoldenSet,
    cases: list[GoldenCase],
    *,
    arm: str,
    base_url: str,
    api_key: str,
    run_id: str,
    repo_root: Path,
    pace_seconds: float,
) -> ArmReport:
    evaluations: list[CaseEvaluation] = []
    skip_rest = False
    skip_reason = None
    for index, case in enumerate(cases):
        if skip_rest:
            outcome = normalize_graph_state(
                None,
                skipped=True,
                skip_reason=skip_reason,
            )
            evaluation = evaluate_case(case, outcome, arm=arm)
            evaluations.append(evaluation)
            continue
        transcript = (repo_root / case.transcript_path).read_text(encoding="utf-8")
        if arm == "rag":
            outcome = run_rag_case(
                client,
                case,
                transcript,
                base_url=base_url,
                api_key=api_key,
                run_id=run_id,
            )
        else:
            outcome = run_graph_case(
                client,
                case,
                transcript,
                base_url=base_url,
                api_key=api_key,
                run_id=run_id,
            )
            if outcome.skipped and outcome.skip_reason == "runtime_unavailable":
                skip_rest = True
                skip_reason = "runtime_unavailable"
        evaluation = evaluate_case(case, outcome, arm=arm)
        evaluations.append(evaluation)
        _print_case(case, evaluation)
        if index < len(cases) - 1 and pace_seconds > 0 and not skip_rest:
            time.sleep(pace_seconds)
    return aggregate(
        evaluations,
        golden.cases,
        arm=arm,  # type: ignore[arg-type]
        skipped=skip_rest and all(ev.verdict == "skipped" for ev in evaluations),
        skip_reason=skip_reason,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden",
        type=Path,
        default=REPO_ROOT / "evals" / "golden_set_s16.json",
    )
    parser.add_argument(
        "--arm",
        choices=("rag", "graph", "both"),
        default="rag",
    )
    parser.add_argument("--case", action="append", dest="case_ids", default=None)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "evals" / "reports")
    parser.add_argument("--label", default=None)
    parser.add_argument("--pace-seconds", type=float, default=7.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the golden set without calling the service or spending tokens.",
    )
    args = parser.parse_args(argv)

    golden_path = args.golden if args.golden.is_absolute() else REPO_ROOT / args.golden
    if not golden_path.is_file():
        print(f"golden set not found: {golden_path}", file=sys.stderr)
        return 2
    golden = load_golden_set(golden_path)
    errors = validate_golden_set(golden, repo_root=REPO_ROOT)
    if errors:
        print("golden set invalid:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    print(
        f"golden set ok: {len(golden.cases)} cases, "
        f"{sum(1 for case in golden.cases if case.expect_abstention)} abstention, "
        f"sha256={_sha256(golden_path)[:12]}…"
    )
    if args.dry_run:
        return 0

    api_key = (
        args.api_key
        or os.environ.get("ESTIMATE_API_KEY")
        or os.environ.get("AI_SERVICE_TOKEN")
    )
    if not api_key:
        print(
            "Provide --api-key or set ESTIMATE_API_KEY / AI_SERVICE_TOKEN",
            file=sys.stderr,
        )
        return 2

    selected = golden.cases
    if args.case_ids:
        wanted = set(args.case_ids)
        selected = [case for case in golden.cases if case.id in wanted]
        missing = wanted - {case.id for case in selected}
        if missing:
            print(f"unknown case ids: {sorted(missing)}", file=sys.stderr)
            return 2

    run_id = uuid.uuid4().hex[:8]
    started = _now()
    arms_to_run = ("rag", "graph") if args.arm == "both" else (args.arm,)
    print(
        f"run_id={run_id} arms={','.join(arms_to_run)} "
        f"cases={[case.id for case in selected]}"
    )
    print("This spends real LLM tokens. Ctrl-C to abort.")

    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=args.timeout) as client:
        environment = fetch_environment(client, base_url=args.base_url, api_key=api_key)
        arm_reports: dict[str, ArmReport] = {}
        for arm in arms_to_run:
            print(f"\n== arm {arm} ==")
            arm_reports[arm] = run_arm(
                client,
                golden,
                selected,
                arm=arm,
                base_url=args.base_url,
                api_key=api_key,
                run_id=run_id,
                repo_root=REPO_ROOT,
                pace_seconds=args.pace_seconds,
            )

    finished = _now()
    ab = None
    if "rag" in arm_reports and "graph" in arm_reports:
        ab = ab_compare(arm_reports["rag"], arm_reports["graph"])
    report = EvalReport(
        run_id=run_id,
        label=args.label,
        started_at=started,
        finished_at=finished,
        golden_set={
            "path": str(golden_path.relative_to(REPO_ROOT)),
            "sha256": _sha256(golden_path),
            "cases": len(golden.cases),
            "selected": [case.id for case in selected],
        },
        environment=environment,
        arms=arm_reports,
        ab=ab,
    )
    stamp = started.strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"eval_s16_{stamp}.json"
    md_path = out_dir / f"eval_s16_{stamp}.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
