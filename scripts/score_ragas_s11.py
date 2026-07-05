#!/usr/bin/env python3
"""Isolated RAGAS scorer for pre-collected Session 11 generation samples.

Imports only ragas + langchain-openai + datasets — never the application stack.
Input: JSON file from ``eval_generation_s11.py --collect-only``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ragas_compat import patch_vertexai_imports  # noqa: E402

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def _build_dataset(rows: list[dict], *, response_key: str = "answer"):
    from ragas.dataset_schema import EvaluationDataset

    samples = [
        {
            "user_input": row["question"],
            "response": row[response_key],
            "retrieved_contexts": row.get("contexts", []),
            "reference": row.get("ground_truth", ""),
        }
        for row in rows
    ]
    return EvaluationDataset.from_list(samples)


def score_rows(
    rows: list[dict], *, openai_api_key: str, embedding_model: str
) -> list[dict]:
    patch_vertexai_imports()

    from openai import OpenAI
    from ragas import evaluate
    from ragas.metrics import (
        AnswerRelevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from ragas.llms import llm_factory
    from ragas.llms.base import LangchainLLMWrapper

    client = OpenAI(api_key=openai_api_key)
    judge_model = "gpt-4o-mini"
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=embedding_model, api_key=openai_api_key)
    )
    llm = llm_factory(judge_model, provider="openai", client=client, max_tokens=16_384)
    relevancy_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=judge_model, api_key=openai_api_key, max_tokens=1024, temperature=0.2
        )
    )

    core = evaluate(
        _build_dataset(rows),
        metrics=[faithfulness, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    relevancy = evaluate(
        _build_dataset(rows),
        metrics=[AnswerRelevancy(strictness=3)],
        llm=relevancy_llm,
        embeddings=embeddings,
    )

    merged: list[dict] = []
    for a, b in zip(core.scores, relevancy.scores, strict=True):
        merged.append({**a, **b})
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score pre-collected RAGAS samples (S11)."
    )
    parser.add_argument(
        "score_file", type=Path, help="JSON samples from --collect-only"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Write metrics JSON here"
    )
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    args = parser.parse_args()

    import os

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required.", file=sys.stderr)
        raise SystemExit(1)

    rows = json.loads(args.score_file.read_text(encoding="utf-8"))
    scores = score_rows(
        rows, openai_api_key=api_key, embedding_model=args.embedding_model
    )

    payload = {"samples": rows, "scores": scores, "metrics": METRICS}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
