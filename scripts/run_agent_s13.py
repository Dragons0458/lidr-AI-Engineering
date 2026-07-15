#!/usr/bin/env python3
"""Session 13 — call POST /v1/estimate/agent/graph with a transcript.

This is an HTTP client of the real endpoint, not a second graph implementation.

    # Against a running API (Compose or uvicorn)
    uv run python scripts/run_agent_s13.py \\
        exercises/session-12/sample_transcript_complex.txt \\
        --base-url http://localhost:8000 \\
        --api-key "$ESTIMATE_API_KEY" \\
        --out exercises/session-13/example_graph_response.json

Evidence checklist after a successful run:
  * response has components, total_hours, confidence, status
  * Logfire shows five ``agent.graph.*`` node spans for the estimation_id
  * Postgres checkpointer has a thread_id equal to estimation_id
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_payload(transcript: str, estimation_id: str | None) -> dict:
    return {
        "estimation_id": estimation_id or f"s13-{uuid.uuid4()}",
        "transcript": transcript,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path, help="Path to the meeting transcript")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="FastAPI base URL",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Value for X-API-Key (defaults to ESTIMATE_API_KEY env via empty→error)",
    )
    parser.add_argument(
        "--estimation-id",
        default=None,
        help="Stable thread_id; generated when omitted",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the JSON response",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="HTTP timeout in seconds",
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
    url = f"{args.base_url.rstrip('/')}/v1/estimate/agent/graph"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    with httpx.Client(timeout=args.timeout) as client:
        response = client.post(url, headers=headers, json=payload)

    print(f"estimation_id={payload['estimation_id']}")
    print(f"HTTP {response.status_code}")
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        print(response.text)
        return 1

    print(json.dumps(body, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "estimation_id": payload["estimation_id"],
                    "status_code": response.status_code,
                    "response": body,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0 if response.is_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
