#!/usr/bin/env python

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("cosine similarity is undefined for zero vectors")
    return dot / (norm_a * norm_b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two texts with OpenAI embeddings and cosine similarity."
    )
    parser.add_argument("--text-a", required=True)
    parser.add_argument("--text-b", required=True)
    return parser.parse_args()


def main() -> None:
    from app.embedding_pipeline.embedder import OpenAIEmbedder

    args = parse_args()
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required in the environment or .env")

    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedder = OpenAIEmbedder(api_key=api_key, model=model)
    vector_a = embedder.embed_one(args.text_a)
    vector_b = embedder.embed_one(args.text_b)
    similarity = cosine_similarity(vector_a, vector_b)

    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Cosine similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
