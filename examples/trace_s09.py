#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27", "openai>=1.40"]
# ///
"""Session 09 pre-work: manual RAG trace over a raw meeting transcript.

This is a client script only: it makes the two calls the trace needs and prints
the raw output. It adds no new behaviour to the estimator service.

Usage, from the repository root:

    docker compose up -d postgres redis api
    docker compose run --rm api python scripts/query_examples.py --ingest
    export OPENAI_API_KEY=sk-...
    uv run examples/trace_s09.py examples/transcripts/02_ambiguous.txt

The base URL is taken from ``ESTIMATOR_BASE_URL`` if set; otherwise
``http://localhost:8000``.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import httpx
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TRANSCRIPT = "examples/transcripts/02_ambiguous.txt"
TOP_K = 5
CONTENT_PREVIEW_CHARS = 140


def base_url() -> str:
    return os.environ.get("ESTIMATOR_BASE_URL", "http://localhost:8000").rstrip("/")


def embed_transcript(text: str) -> list[float]:
    """Embed the whole transcript with the same model used at ingest time."""
    client = OpenAI()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def print_vector_stats(transcript_path: Path, vector: list[float]) -> None:
    norm = math.sqrt(sum(component * component for component in vector))
    print("=" * 78)
    print("STEP 1 - embedding of the full transcript")
    print("=" * 78)
    print(f"transcript      : {transcript_path}")
    print(f"model           : {EMBEDDING_MODEL}")
    print(f"dimensionality  : {len(vector)}")
    print(f"L2 norm         : {norm:.6f}")
    print(f"first component : {vector[0]:.6f}")
    print(f"last component  : {vector[-1]:.6f}")
    print()


def search(text: str) -> dict:
    with httpx.Client(timeout=120.0) as client:
        response = client.post(f"{base_url()}/search", json={"query": text, "k": TOP_K})
        response.raise_for_status()
        return response.json()


def print_search(body: dict) -> None:
    print("=" * 78)
    print(f"STEP 2 - POST /search  (k={TOP_K}, query = full transcript)")
    print("=" * 78)
    print(f"search_time_ms  : {body['search_time_ms']}")
    print(f"results         : {len(body['results'])}")
    print()
    print(
        f"  {'rank':>4}  {'chunk_id':>8}  {'distance':>8}  {'sector':<12}  {'budget_id':<14}  content"
    )
    for rank, hit in enumerate(body["results"], start=1):
        meta = hit.get("metadata", {}) or {}
        sector = str(meta.get("client_sector", meta.get("sector", "?")))
        budget_id = str(meta.get("budget_id", "?"))
        preview = " ".join(hit["content"].split())[:CONTENT_PREVIEW_CHARS]
        print(
            f"  {rank:>4}  {hit['chunk_id']:>8}  {hit['distance']:>8.4f}  "
            f"{sector:<12}  {budget_id:<14}  {preview}"
        )
    print()
    print("--- raw JSON ---")
    print(json.dumps(body, ensure_ascii=False, indent=2))


def main(argv: list[str]) -> int:
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: set OPENAI_API_KEY in the environment.", file=sys.stderr)
        return 1

    transcript_path = Path(argv[1]) if len(argv) > 1 else Path(DEFAULT_TRANSCRIPT)
    if not transcript_path.is_file():
        print(f"ERROR: transcript not found: {transcript_path}", file=sys.stderr)
        return 1

    text = transcript_path.read_text(encoding="utf-8")
    vector = embed_transcript(text)
    print_vector_stats(transcript_path, vector)
    print_search(search(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
