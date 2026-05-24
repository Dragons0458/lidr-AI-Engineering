from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path
from typing import Any

import httpx

from evals.stress.fixtures.build_pdfs import write_fixture_pdfs
from evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
)
from evals.stress.scenarios import SCENARIO_LENGTHS, get_scenario


API_PREFIX = "/api/v1"
DEFAULT_OUTPUT = Path("evals/stress/results.csv")
TURN_OBSERVED_FIELDS = [
    "turn_index",
    "session_id",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
]
CSV_FIELDS = [
    "scenario",
    "scenario_base",
    "scenario_turns",
    "attachment_size_kb",
    "repeat",
    "expected_fact",
    *TURN_OBSERVED_FIELDS,
    "latency_budget_passed",
    "cost_budget_passed",
    "memory_drift_passed",
    "memory_drift_score",
]


async def run_stress(
    *,
    http_base_url: str | None,
    scenario_names: list[str],
    attachment_sizes_kb: list[int],
    repeats: int,
    output_path: Path,
    turn_counts: list[int],
    latency_budget_ms: int,
    cost_budget_usd: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_paths = _fixture_paths()
    rows: list[dict[str, Any]] = []

    async with _stress_client(http_base_url) as client:
        for scenario_name in scenario_names:
            for turn_count in turn_counts:
                for attachment_size_kb in attachment_sizes_kb:
                    for repeat in range(1, repeats + 1):
                        session_id = await _create_session(client)
                        rows.extend(
                            await _run_session(
                                client=client,
                                session_id=session_id,
                                scenario_base=scenario_name,
                                scenario_turns=turn_count,
                                attachment_size_kb=attachment_size_kb,
                                attachment_path=fixture_paths.get(attachment_size_kb),
                                repeat=repeat,
                                latency_budget_ms=latency_budget_ms,
                                cost_budget_usd=cost_budget_usd,
                            )
                        )

    _write_csv(output_path, rows)


async def _run_session(
    *,
    client: httpx.AsyncClient,
    session_id: str,
    scenario_base: str,
    scenario_turns: int,
    attachment_size_kb: int,
    attachment_path: Path | None,
    repeat: int,
    latency_budget_ms: int,
    cost_budget_usd: float,
) -> list[dict[str, Any]]:
    scenario = get_scenario(scenario_base, max_turns=scenario_turns)
    latency_metric = LatencyBudgetMetric(latency_budget_ms)
    cost_metric = CostBudgetMetric(cost_budget_usd)
    rows = []

    for turn in scenario.turns:
        await _post_estimate(
            client=client,
            session_id=session_id,
            transcript=turn.transcript,
            attachment_path=attachment_path,
        )
        snapshot = await _get_session_snapshot(client, session_id)
        observation = snapshot.get("last_turn_observation") or {}
        memory_score, memory_passed = _evaluate_memory(
            facts=scenario.facts_before(turn.turn_index),
            snapshot=snapshot,
        )

        row = {
            "scenario": scenario.name,
            "scenario_base": scenario_base,
            "scenario_turns": scenario_turns,
            "attachment_size_kb": attachment_size_kb,
            "repeat": repeat,
            "expected_fact": turn.fact_to_remember,
            **{field: observation.get(field, "") for field in TURN_OBSERVED_FIELDS},
            "latency_budget_passed": int(latency_metric.evaluate(observation).passed),
            "cost_budget_passed": int(cost_metric.evaluate(observation).passed),
            "memory_drift_passed": int(memory_passed),
            "memory_drift_score": memory_score,
        }
        rows.append(row)

    return rows


async def _post_estimate(
    *,
    client: httpx.AsyncClient,
    session_id: str,
    transcript: str,
    attachment_path: Path | None,
) -> None:
    data = {
        "description": transcript,
        "project_type": "web_saas",
        "detail_level": "medium",
        "output_format": "line_items",
        "evaluate": "false",
    }

    files = None
    if attachment_path is not None:
        files = {
            "attachments": (
                attachment_path.name,
                attachment_path.read_bytes(),
                "application/pdf",
            )
        }

    response = await client.post(
        f"{API_PREFIX}/sessions/{session_id}/estimate",
        data=data,
        files=files,
    )
    response.raise_for_status()


async def _create_session(client: httpx.AsyncClient) -> str:
    response = await client.post(f"{API_PREFIX}/sessions")
    response.raise_for_status()
    return str(response.json()["session_id"])


async def _get_session_snapshot(
    client: httpx.AsyncClient, session_id: str
) -> dict[str, Any]:
    response = await client.get(f"{API_PREFIX}/sessions/{session_id}")
    response.raise_for_status()
    return response.json()


def _evaluate_memory(
    *, facts: tuple[str, ...], snapshot: dict[str, Any]
) -> tuple[float, bool]:
    if not facts:
        return 1.0, True

    results = [MemoryDriftMetric(fact).evaluate(snapshot) for fact in facts]
    score = sum(result.score for result in results) / len(results)
    return round(score, 3), all(result.passed for result in results)


def _fixture_paths() -> dict[int, Path | None]:
    paths = write_fixture_pdfs()
    return {
        0: None,
        **{
            int(path.stem.removeprefix("attach_").removesuffix("kb")): path
            for path in paths
        },
    }


def _write_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _stress_client(http_base_url: str | None) -> httpx.AsyncClient:
    if http_base_url:
        return httpx.AsyncClient(base_url=http_base_url.rstrip("/"), timeout=120)

    from app.main import app

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        timeout=120,
    )


def _parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv_arg(value: str) -> list[int]:
    return [int(item) for item in _parse_csv_arg(value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CAG stress scenarios.")
    parser.add_argument(
        "--http", default=None, help="Base URL, e.g. http://localhost:8000"
    )
    parser.add_argument(
        "--scenarios",
        default="growing,pivot,contradiction",
        help="Comma-separated scenario names.",
    )
    parser.add_argument(
        "--attachment-sizes",
        default="0,5,20,50,100",
        help="Comma-separated synthetic PDF sizes in KB. Use 0 for no attachment.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--turn-counts",
        default=",".join(str(count) for count in SCENARIO_LENGTHS),
        help="Comma-separated scenario truncation lengths.",
    )
    parser.add_argument("--latency-budget-ms", type=int, default=4000)
    parser.add_argument("--cost-budget-usd", type=float, default=0.05)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    await run_stress(
        http_base_url=args.http,
        scenario_names=_parse_csv_arg(args.scenarios),
        attachment_sizes_kb=_parse_int_csv_arg(args.attachment_sizes),
        repeats=args.repeats,
        output_path=args.output,
        turn_counts=_parse_int_csv_arg(args.turn_counts),
        latency_budget_ms=args.latency_budget_ms,
        cost_budget_usd=args.cost_budget_usd,
    )


if __name__ == "__main__":
    asyncio.run(main())
