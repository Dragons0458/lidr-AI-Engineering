from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from scripts import run_agent_s14 as cli


def test_build_payload_preserves_id_and_generates_one_when_missing() -> None:
    assert cli.build_payload("brief", "estimate-14") == {
        "estimation_id": "estimate-14",
        "transcript": "brief",
    }

    generated = cli.build_payload("brief", None)

    assert generated["estimation_id"].startswith("s14-")
    assert generated["transcript"] == "brief"


def test_http_flow_captures_review_and_reads_final_checkpoint() -> None:
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, body))
        if request.url.path.endswith("/resume"):
            return httpx.Response(200, json={"state": "completed"})
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"state": "completed", "status": "validated"},
            )
        return httpx.Response(
            200,
            json={
                "state": "paused",
                "pending_review": {
                    "reasons": ["high_risk_scope"],
                    "risk_flags": ["novel_cryptography"],
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        evidence = cli.run_http_flow(
            client,
            base_url="http://testserver/",
            api_key="secret",
            payload={"estimation_id": "estimate-14", "transcript": "brief"},
            decision="approve",
        )

    assert evidence.state == {"state": "completed", "status": "validated"}
    assert evidence.review_triggered is True
    assert evidence.review_reasons == ["high_risk_scope"]
    assert evidence.risk_flags == ["novel_cryptography"]
    assert evidence.decision == "approve"
    assert requests == [
        (
            "POST",
            "/v1/estimate/agent/supervisor",
            {"estimation_id": "estimate-14", "transcript": "brief"},
        ),
        (
            "POST",
            "/v1/estimate/agent/supervisor/estimate-14/resume",
            {
                "decision": "approve",
                "note": "auto-decided by run_agent_s14.py",
            },
        ),
        (
            "GET",
            "/v1/estimate/agent/supervisor/estimate-14/state",
            None,
        ),
    ]


@pytest.mark.asyncio
async def test_generate_evidence_is_reproducible_and_covers_three_cases(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_paths = await cli.generate_evidence(first_dir)
    second_paths = await cli.generate_evidence(second_dir)

    assert [path.name for path in first_paths] == [
        "example_run_happy.txt",
        "example_run_edge_case.txt",
        "example_run_violate.txt",
    ]
    first = {path.name: path.read_text(encoding="utf-8") for path in first_paths}
    second = {path.name: path.read_text(encoding="utf-8") for path in second_paths}
    assert first == second

    assert "estimation_id = s14-evidence-happy" in first["example_run_happy.txt"]
    assert "triggered: no" in first["example_run_happy.txt"]
    assert "status = validated" in first["example_run_happy.txt"]
    assert "triggered: YES" in first["example_run_edge_case.txt"]
    assert "high_risk_scope" in first["example_run_edge_case.txt"]
    assert "novel_cryptography" in first["example_run_edge_case.txt"]
    assert "[DENIED]" in first["example_run_violate.txt"]
    assert "tool:validate_estimate" in first["example_run_violate.txt"]
