#!/usr/bin/env python3
"""Run five semantic search queries against POST /embeddings/search.

Assumes the API is reachable and budgets from ``data/budgets_sample.json`` are
already ingested (one POST /embeddings/ingest per budget). Use ``--ingest`` to
ingest the corpus before searching.

Usage:
    docker compose run --rm api python scripts/query_examples.py
    docker compose run --rm api python scripts/query_examples.py --ingest
    uv run python scripts/query_examples.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

QUERIES = [
    (
        "Component direct match (sanity check)",
        "REST API development with JWT authentication for financial sector",
    ),
    (
        "Semantic reformulation",
        "secure backend service with token-based access control for banking applications",
    ),
    (
        "Out-of-domain query",
        "mobile application for restaurant reservations",
    ),
    (
        "Ambiguous short query",
        "integration with external system",
    ),
    (
        "Highly specific technical query",
        "migration from monolith to microservices architecture using Kubernetes",
    ),
]

CONTENT_PREVIEW_CHARS = 120
DEFAULT_BASE_URL = "http://localhost:8000"
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "budgets_sample.json"


def ingest_corpus(client: httpx.Client, budgets: list[dict]) -> None:
    for budget in budgets:
        source_path = f"data/budgets/{budget['budget_id'].lower()}.json"
        payload = {
            "source_path": source_path,
            "document_type": "historical_budget",
            "content": budget,
        }
        response = client.post("/embeddings/ingest", json=payload, timeout=120.0)
        if response.status_code == 409:
            print(f"  skip (already ingested): {source_path}")
            continue
        response.raise_for_status()
        body = response.json()
        print(
            f"  ingested {source_path}: "
            f"document_id={body['document_id']}, chunks={body['chunks_created']}"
        )


def print_search_results(
    client: httpx.Client, label: str, query: str, k: int = 5
) -> None:
    print(f"\n{'=' * 72}")
    print(f"Query: {label}")
    print(f"Text:  {query}")
    print("-" * 72)

    response = client.post(
        "/embeddings/search",
        json={"query": query, "k": k},
        timeout=60.0,
    )
    response.raise_for_status()
    body = response.json()
    print(f"search_time_ms={body['search_time_ms']}, hits={len(body['results'])}")

    for rank, hit in enumerate(body["results"], start=1):
        preview = hit["content"].replace("\n", " ")[:CONTENT_PREVIEW_CHARS]
        if len(hit["content"]) > CONTENT_PREVIEW_CHARS:
            preview += "..."
        print(
            f"  {rank}. chunk_id={hit['chunk_id']}  "
            f"distance={hit['distance']:.4f}  "
            f"chunk_type={hit['chunk_type']}"
        )
        print(f"     {preview}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Semantic search query examples (Session 8)"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest data/budgets_sample.json before running queries",
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url.rstrip("/")) as client:
        health = client.get("/health", timeout=10.0)
        health.raise_for_status()

        if args.ingest:
            budgets = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
            print(f"Ingesting {len(budgets)} budgets from {SAMPLE_PATH}...")
            ingest_corpus(client, budgets)

        for label, query in QUERIES:
            print_search_results(client, label, query, k=args.k)

    return 0


if __name__ == "__main__":
    sys.exit(main())
