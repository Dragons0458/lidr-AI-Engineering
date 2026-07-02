#!/usr/bin/env python3
"""Offline RAGAS evaluation over the Session 11 generation golden set.

For each case in ``evals/golden_generation.json`` the script runs the real
grounded pipeline (``estimate_from_transcript``), collects the four RAGAS
columns (question, answer, contexts, ground_truth) and prints a Markdown table
with faithfulness, answer_relevancy, context_precision and context_recall.

Usage (stack up + corpus ingested + OPENAI_API_KEY in .env)::

    uv sync --group dev
    uv run python scripts/eval_ragas_s11.py

Optional flags::

    uv run python scripts/eval_ragas_s11.py --cache results/ragas_cache.json
    uv run python scripts/eval_ragas_s11.py --metrics-only --cache results/ragas_cache.json

The ``--cache`` file stores generated answers/contexts so metric re-runs skip
the expensive generation step.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ragas_compat import patch_vertexai_imports  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.generation.rag.estimator import estimate_from_transcript  # noqa: E402
from app.generation.rag.serialization import (  # noqa: E402
    compact_response_for_relevancy,
    render_estimate_as_text,
)

GOLDEN_PATH = ROOT / "evals" / "golden_generation.json"
METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


async def _run_pipeline(question: str) -> tuple[str, list[str]]:
    """Return (answer text, retrieved context strings) for one golden case."""
    from app.dependencies import get_embedder, get_token_encoder
    from app.generation.rag.context_assembler import (
        build_context_block,
        truncate_to_token_budget,
    )
    from app.generation.rag.query_reformulator import (
        compose_search_text,
        reformulate_query,
    )
    from app.generation.rag.retrieval.pipeline import retrieve
    from app.dependencies import get_runtime_retrieval_config

    settings = get_settings()
    query = await reformulate_query(question)
    search_text = compose_search_text(query)
    embedder = get_embedder()
    if embedder is None:
        raise RuntimeError("Embedding service unavailable (set OPENAI_API_KEY).")
    query_embedding = await asyncio.to_thread(embedder.embed_one, search_text)

    runtime_retrieval = get_runtime_retrieval_config()
    retrieval = await retrieve(
        query_embedding=query_embedding,
        query_text=search_text,
        search_mode=runtime_retrieval.effective_search_mode(),
        rerank=runtime_retrieval.effective_rerank(),
        top_k=settings.RETRIEVAL_TOP_K,
        recall_k=settings.RETRIEVAL_RECALL_TOP_K,
        rerank_top_n=settings.RERANK_TOP_N,
        distance_threshold=settings.RETRIEVAL_DISTANCE_THRESHOLD,
        rrf_k=settings.RRF_K,
    )

    if retrieval.low_confidence:
        estimate = await estimate_from_transcript(question)
        return render_estimate_as_text(estimate), []

    encoder = get_token_encoder()
    kept = truncate_to_token_budget(
        retrieval.chunks, settings.MAX_CONTEXT_TOKENS, encoder
    )
    contexts = [chunk.content for chunk in kept]
    context_block = build_context_block(kept)

    from app.generation.rag.estimator import generate_estimate

    estimate = await generate_estimate(context_block, structured_query=query)
    from app.generation.rag.validation import (
        verify_citations,
        degrade_dangling_tasks,
        log_citation_report,
    )

    report = verify_citations(estimate, kept)
    log_citation_report(report, request_id=f"eval-{question[:24]}")
    if report.has_dangling:
        estimate = degrade_dangling_tasks(estimate, report)

    return render_estimate_as_text(estimate), contexts


def _load_cache(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_cache(path: Path | None, data: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _ragas_judge_model(settings) -> str:
    """Cheap OpenAI judge for RAGAS (independent of generation model)."""
    model = settings.PRIMARY_MODEL or "gpt-4o-mini"
    if model.startswith("gpt-"):
        return model
    return "gpt-4o-mini"


def _build_dataset(rows: list[dict], *, response_key: str = "answer"):
    from ragas.dataset_schema import EvaluationDataset

    samples = [
        {
            "user_input": row["question"],
            "response": row[response_key],
            "retrieved_contexts": row["contexts"],
            "reference": row["ground_truth"],
        }
        for row in rows
    ]
    return EvaluationDataset.from_list(samples)


def _judge_llm(settings, client):
    """Instructor LLM for structured metrics (faithfulness, context_*)."""
    from ragas.llms import llm_factory

    return llm_factory(
        _ragas_judge_model(settings),
        provider="openai",
        client=client,
        max_tokens=16_384,
    )


def _relevancy_llm(settings):
    """LangChain LLM so RAGAS can batch ``strictness`` parallel generations."""
    from langchain_openai import ChatOpenAI
    from ragas.llms.base import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatOpenAI(
            model=_ragas_judge_model(settings),
            api_key=settings.OPENAI_API_KEY,
            max_tokens=1_024,
            temperature=0.2,
        )
    )


def _langchain_embeddings(settings):
    from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
    from ragas.embeddings.base import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(
        LangchainOpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )
    )


def _merge_scores(*score_lists: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for rows in zip(*score_lists, strict=True):
        combined: dict = {}
        for part in rows:
            combined.update(part)
        merged.append(combined)
    return merged


def _evaluate(rows: list[dict]) -> list[dict]:
    patch_vertexai_imports()

    from openai import OpenAI
    from ragas import evaluate
    from ragas.metrics import (
        AnswerRelevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for RAGAS judge + embeddings.")

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    embeddings = _langchain_embeddings(settings)

    # Core metrics need the full grounded answer (claims + citations).
    core_result = evaluate(
        _build_dataset(rows),
        metrics=[faithfulness, context_precision, context_recall],
        llm=_judge_llm(settings, client),
        embeddings=embeddings,
    )

    # answer_relevancy: compact scope summary + LangChain batched generations.
    relevancy_rows = [
        {**row, "relevancy_answer": compact_response_for_relevancy(row["answer"])}
        for row in rows
    ]
    relevancy_metric = AnswerRelevancy(strictness=3)
    relevancy_result = evaluate(
        _build_dataset(relevancy_rows, response_key="relevancy_answer"),
        metrics=[relevancy_metric],
        llm=_relevancy_llm(settings),
        embeddings=embeddings,
    )

    return _merge_scores(core_result.scores, relevancy_result.scores)


def _print_table(rows: list[dict], scores: list[dict]) -> None:
    print("\n## RAGAS generation evaluation (Session 11)\n")
    header = "| Case | Category | " + " | ".join(METRICS) + " |"
    print(header)
    print("| --- | --- | " + " | ".join("---" for _ in METRICS) + " |")

    # Context precision/recall compare against the reference answer; for an
    # abstention case the correct reference has no factual claims to retrieve, so
    # a 0 there is structural, not a quality signal. Report it as n/a and keep it
    # out of the averages instead of dragging them down.
    abstention_na_metrics = {"context_precision", "context_recall"}

    metric_avgs: dict[str, list[float]] = {m: [] for m in METRICS}
    for case, score in zip(rows, scores, strict=True):
        cells = []
        is_abstention = case.get("category") == "abstention"
        for metric in METRICS:
            value = score.get(metric)
            if is_abstention and metric in abstention_na_metrics:
                cells.append("n/a")
            elif value is None:
                cells.append("n/a")
            else:
                cells.append(f"{float(value):.3f}")
                metric_avgs[metric].append(float(value))
        print(
            f"| {case['id']} | {case.get('category', '')} | " + " | ".join(cells) + " |"
        )

    avg_cells = []
    for metric in METRICS:
        values = metric_avgs[metric]
        avg_cells.append(f"{statistics.fmean(values):.3f}" if values else "n/a")
    print("| **AVG** | — | " + " | ".join(avg_cells) + " |")


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="RAGAS eval over golden_generation.json"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "evals" / "ragas_cache.json",
        help="JSON file to cache generated answers/contexts between runs.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip generation; require a populated --cache file.",
    )
    args = parser.parse_args()

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    cases = golden["cases"]
    cache = _load_cache(args.cache)
    rows: list[dict] = []

    for case in cases:
        case_id = case["id"]
        cached = cache.get(case_id)
        if args.metrics_only:
            if not cached:
                print(f"Missing cache entry for {case_id}", file=sys.stderr)
                return 1
            rows.append({**case, **cached})
            continue

        if cached and "answer" in cached and "contexts" in cached:
            answer, contexts = cached["answer"], cached["contexts"]
            print(f"[{case_id}] using cached generation")
        else:
            print(f"[{case_id}] running grounded pipeline...")
            answer, contexts = await _run_pipeline(case["question"])
            cache[case_id] = {"answer": answer, "contexts": contexts}
            _save_cache(args.cache, cache)

        rows.append(
            {
                "id": case_id,
                "category": case.get("category"),
                "question": case["question"],
                "ground_truth": case["ground_truth"],
                "answer": answer,
                "contexts": contexts,
            }
        )

    print("Running RAGAS metrics (LLM judge — may take several minutes)...")
    scores = _evaluate(rows)
    _print_table(rows, scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
