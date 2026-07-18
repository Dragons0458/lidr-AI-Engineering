#!/usr/bin/env python3
"""Session 13 — HTTP client for the multi-agent graph (start → resume → state).

    uv run python scripts/run_agent_s13.py \\
        exercises/session-12/sample_transcript_complex.txt \\
        --base-url http://localhost:8000 \\
        --api-key "$ESTIMATE_API_KEY" \\
        --out exercises/session-13/example_graph_response.json

Evidence checklist after a successful run:
  * run pauses at gate 1 (structure_review), then gate 2 (final_review)
  * final state has status=validated and optional proposal markdown
  * Postgres checkpointer has thread_id equal to estimation_id
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PREFIX = "/v1/estimate/agent/graph"


def build_payload(transcript: str, estimation_id: str | None) -> dict:
    return {
        "estimation_id": estimation_id or f"s13-{uuid.uuid4()}",
        "transcript": transcript,
    }


def canned_decision(gate: str, *, want_proposal: bool) -> dict:
    if gate == "structure_review":
        return {"approved": True}
    if gate == "final_review":
        return {"validated": True, "want_proposal": want_proposal}
    raise ValueError(f"Unknown gate: {gate!r}")


def run_flow(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    payload: dict,
    want_proposal: bool,
) -> dict:
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    root = base_url.rstrip("/")
    estimation_id = payload["estimation_id"]

    response = client.post(f"{root}{GRAPH_PREFIX}", headers=headers, json=payload)
    response.raise_for_status()
    state = response.json()

    while state.get("state") == "paused":
        pending = state.get("pending_gate") or {}
        gate = pending.get("gate", "unknown")
        decision = canned_decision(gate, want_proposal=want_proposal)
        resume = client.post(
            f"{root}{GRAPH_PREFIX}/{estimation_id}/resume",
            headers=headers,
            json={"decision": decision},
        )
        resume.raise_for_status()
        state = resume.json()

    final = client.get(
        f"{root}{GRAPH_PREFIX}/{estimation_id}/state",
        headers=headers,
    )
    final.raise_for_status()
    return final.json()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path, help="Path to the meeting transcript")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--estimation-id", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--no-proposal",
        action="store_true",
        help="Skip the bonus proposal at gate 2",
    )
    args = parser.parse_args(argv)

    transcript_path = args.transcript
    if not transcript_path.is_file():
        print(f"Transcript not found: {transcript_path}", file=sys.stderr)
        return 2

    api_key = args.api_key
    if not api_key:
        import os

        api_key = os.environ.get("ESTIMATE_API_KEY")
    if not api_key:
        print("Provide --api-key or set ESTIMATE_API_KEY", file=sys.stderr)
        return 2

    payload = build_payload(
        transcript_path.read_text(encoding="utf-8"), args.estimation_id
    )
    print(f"estimation_id={payload['estimation_id']}")

    try:
        with httpx.Client(timeout=args.timeout) as client:
            state = run_flow(
                client,
                base_url=args.base_url,
                api_key=api_key,
                payload=payload,
                want_proposal=not args.no_proposal,
            )
    except httpx.HTTPStatusError as exc:
        print(f"HTTP {exc.response.status_code}", file=sys.stderr)
        print(exc.response.text, file=sys.stderr)
        return 1

    print(json.dumps(state, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "estimation_id": payload["estimation_id"],
                    "response": state,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
