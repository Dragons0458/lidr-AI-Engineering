#!/usr/bin/env python3
"""Session 15 post-deploy smoke test — shape only, not estimate quality.

POST-DEPLOY ONLY. Do NOT run from CI: this calls the real LLM and spends tokens.
Run from the public ``web`` container so the path under test is
``web → estimador_net → ai-service`` with ``X-API-Key``:

    docker compose -f docker-compose.yml exec web python scripts/smoke_test_s15.py

Requires ESTIMATE_API_KEY / AI_SERVICE_TOKEN and a seeded vector corpus
(see Phase A / docs/deployment-local.md). Use --allow-insufficient only when
you deliberately want a green exit without proving check 4.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

ROOT = os.getenv("AI_SERVICE_URL", "http://localhost:8000").rstrip("/")
TOKEN = (
    os.getenv("AI_SERVICE_TOKEN", "").strip()
    or os.getenv("ESTIMATE_API_KEY", "").strip()
)
IDEMPOTENCY_KEY = "s15-smoke-happy-path"
VALID_CONFIDENCE = frozenset({"high", "medium", "low"})

_CANDIDATE_TRANSCRIPTS = (
    Path(__file__).resolve().parent / "data" / "s15_smoke_transcript.txt",
    Path(__file__).resolve().parents[1]
    / "exercises"
    / "session-14"
    / "sample_transcript_happy_path.txt",
    Path("/app/scripts/data/s15_smoke_transcript.txt"),
    Path("/app/exercises/session-14/sample_transcript_happy_path.txt"),
)


def _load_transcript() -> str:
    for path in _CANDIDATE_TRANSCRIPTS:
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if len(text) >= 100:
                print(f"transcript {path} ({len(text)} chars)")
                return text
    print(
        "FAIL: sample transcript not found "
        "(expected exercises/session-14/sample_transcript_happy_path.txt)",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-insufficient",
        action="store_true",
        help=(
            "Accept confidence=insufficient (empty corpus). "
            "Does NOT count as Paso 7 check 4 evidence."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not TOKEN:
        print("FAIL: set ESTIMATE_API_KEY or AI_SERVICE_TOKEN", file=sys.stderr)
        return 2

    headers = {"X-API-Key": TOKEN}
    with httpx.Client(timeout=180.0) as client:
        health = client.get(f"{ROOT}/health")
        print("health", health.status_code, health.json())
        if health.status_code != 200:
            return 1

        ready = client.get(f"{ROOT}/health/ready")
        print("ready", ready.status_code, ready.json())
        if ready.status_code != 200:
            return 1

        config = client.get(f"{ROOT}/api/v1/config/models", headers=headers)
        print("config/models", config.status_code)
        if config.status_code != 200:
            print(config.text[:400], file=sys.stderr)
            return 1

        unauth = client.get(f"{ROOT}/api/v1/config/models")
        print("config/models (no token)", unauth.status_code)
        if unauth.status_code != 401:
            print("FAIL: expected 401 without token", file=sys.stderr)
            return 1

        transcript = _load_transcript()
        estimate = client.post(
            f"{ROOT}/v1/estimate/from-transcript",
            headers=headers,
            json={
                "transcript": transcript,
                "idempotency_key": IDEMPOTENCY_KEY,
            },
        )
        print("POST /v1/estimate/from-transcript", estimate.status_code)
        if estimate.status_code != 200:
            print(estimate.text[:800], file=sys.stderr)
            print("FAIL: estimate status_code != 200", file=sys.stderr)
            return 1

        body = estimate.json()
        confidence = body.get("confidence")
        sources = body.get("sources") or []
        modules = body.get("modules") or []
        days = body.get("total_engineer_days")

        if confidence == "insufficient":
            msg = (
                "FAIL: confidence=insufficient — corpus likely empty "
                "(run Fase A / build_task_corpus.py --ingest). "
                "This is NOT a frontier/network failure."
            )
            if args.allow_insufficient:
                print(msg.replace("FAIL:", "WARN:"), file=sys.stderr)
                print(
                    "OK — smoke_test_s15 passed with --allow-insufficient "
                    "(NOT valid as check-4 evidence)"
                )
                return 0
            print(msg, file=sys.stderr)
            return 1

        if confidence not in VALID_CONFIDENCE:
            print(
                f"FAIL: confidence={confidence!r} not in {sorted(VALID_CONFIDENCE)}",
                file=sys.stderr,
            )
            return 1

        if not sources:
            print(
                "FAIL: sources is empty — pgvector retrieval did not ground the estimate",
                file=sys.stderr,
            )
            return 1
        for i, src in enumerate(sources):
            sid = src.get("source_id") if isinstance(src, dict) else None
            if not isinstance(sid, int):
                print(
                    f"FAIL: sources[{i}].source_id must be int, got {sid!r}",
                    file=sys.stderr,
                )
                return 1

        if days is None or not isinstance(days, int) or days <= 0:
            print(
                f"FAIL: total_engineer_days must be int > 0, got {days!r}",
                file=sys.stderr,
            )
            return 1

        if not modules:
            print("FAIL: modules is empty", file=sys.stderr)
            return 1

        print(
            f"estimate: confidence={confidence}, sources={len(sources)}, "
            f"days={days}, modules={len(modules)}"
        )

    print(
        "OK — smoke_test_s15 passed "
        f"(estimate: confidence={confidence}, sources={len(sources)}, days={days})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
