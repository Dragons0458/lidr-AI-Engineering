#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two texts with OpenAI embeddings and cosine similarity."
    )
    parser.add_argument("--text-a", required=True)
    parser.add_argument("--text-b", required=True)
    return parser.parse_args()


def main() -> None:
    from openai import OpenAI

    from app.generation.rag.analysis.similarity import cosine_similarity
    from app.generation.rag.embedding.embedder import OpenAIEmbedder

    args = parse_args()
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required in the environment or .env")

    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedder = OpenAIEmbedder(client=OpenAI(api_key=api_key), model=model)
    vector_a = embedder.embed_one(args.text_a)
    vector_b = embedder.embed_one(args.text_b)
    similarity = cosine_similarity(vector_a, vector_b)

    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Cosine similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
