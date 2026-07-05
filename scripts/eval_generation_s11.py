#!/usr/bin/env python3
"""Session 11 generation eval harness — collect, score, gate and compare configs.

Two-phase workflow (same as ``eval_ragas_s11.py``):

    uv run python scripts/eval_generation_s11.py --gate --config full --collect-only s.json
    uv run python scripts/score_ragas_s11.py s.json --output metrics.json
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

from app.config import get_settings  # noqa: E402
from app.generation.rag.estimator import estimate_from_transcript  # noqa: E402
from app.generation.rag.serialization import (  # noqa: E402
    compact_response_for_relevancy,
    render_estimate_as_text,
)
from app.generation.rag.validation import (  # noqa: E402
    degrade_dangling_tasks,
    log_citation_report,
    verify_citations,
)

GOLDEN_PATH = ROOT / "evals" / "golden_generation.json"
BASELINE_PATH = ROOT / "evals" / "ragas_baseline_s11.json"
TOLERANCE = 0.08

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

MONITOR_METRICS = ["faithfulness", "answer_relevancy"]

NAMED_CONFIGS: dict[str, dict[str, bool]] = {
    "full": {
        "augmentation": True,
        "synthesis": True,
        "hallucination_gate": True,
    },
    "no_augment": {
        "augmentation": False,
        "synthesis": True,
        "hallucination_gate": True,
    },
    "no_gate": {
        "augmentation": True,
        "synthesis": True,
        "hallucination_gate": False,
    },
    "minimal": {
        "augmentation": False,
        "synthesis": False,
        "hallucination_gate": False,
    },
}


def _apply_runtime_config(config_name: str) -> None:
    """Toggle S11 flags via Redis-backed runtime config."""
    from app.dependencies import get_runtime_retrieval_config
    from app.foundation.llm.runtime_config import (
        AUGMENTATION_KEY,
        HALLUCINATION_GATE_KEY,
        SYNTHESIS_KEY,
    )

    toggles = NAMED_CONFIGS[config_name]
    runtime = get_runtime_retrieval_config()
    runtime._set_raw(  # noqa: SLF001 — eval harness needs direct override
        AUGMENTATION_KEY, str(toggles["augmentation"]).lower()
    )
    runtime._set_raw(SYNTHESIS_KEY, str(toggles["synthesis"]).lower())
    runtime._set_raw(HALLUCINATION_GATE_KEY, str(toggles["hallucination_gate"]).lower())


async def _collect_case(case: dict) -> dict:
    from app.dependencies import get_embedder, get_token_encoder
    from app.generation.rag.context_assembler import (
        build_context_block,
        truncate_to_token_budget,
    )
    from app.generation.rag.estimator import generate_estimate
    from app.generation.rag.query_reformulator import (
        compose_search_text,
        reformulate_query,
    )
    from app.generation.rag.quality.augmentation import augment_chunks
    from app.generation.rag.retrieval.pipeline import retrieve
    from app.dependencies import get_runtime_retrieval_config

    settings = get_settings()
    question = case["question"]
    query = await reformulate_query(question)
    search_text = compose_search_text(query)
    embedder = get_embedder()
    if embedder is None:
        raise RuntimeError("Embedding service unavailable (set OPENAI_API_KEY).")
    query_embedding = await asyncio.to_thread(embedder.embed_one, search_text)

    runtime = get_runtime_retrieval_config()
    retrieval = await retrieve(
        query_embedding=query_embedding,
        query_text=search_text,
        search_mode=runtime.effective_search_mode(),
        rerank=runtime.effective_rerank(),
        top_k=settings.RETRIEVAL_TOP_K,
        recall_k=settings.RETRIEVAL_RECALL_TOP_K,
        rerank_top_n=settings.RERANK_TOP_N,
        distance_threshold=settings.RETRIEVAL_DISTANCE_THRESHOLD,
        rrf_k=settings.RRF_K,
    )

    if retrieval.low_confidence:
        estimate = await estimate_from_transcript(question)
        return {
            "id": case["id"],
            "category": case.get("category", ""),
            "question": question,
            "ground_truth": case.get("ground_truth", ""),
            "answer": render_estimate_as_text(estimate),
            "contexts": [],
            "dangling_citations": 0,
        }

    encoder = get_token_encoder()
    kept = truncate_to_token_budget(
        retrieval.chunks, settings.MAX_CONTEXT_TOKENS, encoder
    )
    if runtime.effective_augmentation():
        kept = augment_chunks(
            kept,
            compress=settings.AUGMENTATION_COMPRESS,
            reorder=settings.AUGMENTATION_REORDER,
        )
    contexts = [chunk.content for chunk in kept]
    context_block = build_context_block(kept)
    estimate = await generate_estimate(context_block, structured_query=query)

    report = verify_citations(estimate, kept)
    log_citation_report(report, request_id=f"eval-{case['id']}")
    if report.has_dangling:
        estimate = degrade_dangling_tasks(estimate, report)

    if runtime.effective_hallucination_gate():
        from app.generation.rag.quality.hallucination import gate_estimate

        await gate_estimate(
            estimate,
            kept,
            tolerance=settings.HALLUCINATION_NUMERIC_TOLERANCE,
            judge_model=settings.HALLUCINATION_JUDGE_MODEL,
            use_judge=False,
        )

    return {
        "id": case["id"],
        "category": case.get("category", ""),
        "question": question,
        "ground_truth": case.get("ground_truth", ""),
        "answer": render_estimate_as_text(estimate),
        "relevancy_answer": compact_response_for_relevancy(
            render_estimate_as_text(estimate)
        ),
        "contexts": contexts,
        "dangling_citations": len(report.dangling),
    }


async def collect_all(config_name: str) -> list[dict]:
    _apply_runtime_config(config_name)
    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return [await _collect_case(case) for case in cases]


def _avg_metric(scores: list[dict], metric: str) -> float | None:
    values = [
        float(s[metric])
        for s in scores
        if s.get(metric) is not None and s.get(metric) == s.get(metric)
    ]
    if not values:
        return None
    return statistics.fmean(values)


def _gate_check(rows: list[dict], scores: list[dict]) -> bool:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline_avgs = baseline.get("averages", {})
    ok = True

    for metric in METRICS:
        current = _avg_metric(scores, metric)
        base = baseline_avgs.get(metric)
        if current is None or base is None:
            continue
        if current < base - TOLERANCE:
            print(f"FAIL: {metric} {current:.3f} < baseline {base:.3f} - {TOLERANCE}")
            ok = False

    dangling = sum(r.get("dangling_citations", 0) for r in rows)
    if dangling > 0:
        print(f"FAIL: {dangling} dangling citation(s) in collected samples")
        ok = False

    return ok


def _print_scoreboard(config_results: dict[str, list[dict]]) -> None:
    print("\n## Config scoreboard (collect-only)\n")
    print("| Config | dangling | samples |")
    print("| --- | --- | --- |")
    for name, rows in config_results.items():
        dangling = sum(r.get("dangling_citations", 0) for r in rows)
        print(f"| {name} | {dangling} | {len(rows)} |")


async def _run_gate(config_name: str, score_file: Path | None) -> int:
    rows = await collect_all(config_name)
    if score_file:
        score_file.parent.mkdir(parents=True, exist_ok=True)
        score_file.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Wrote {len(rows)} samples to {score_file}")
        print("Score with: uv run python scripts/score_ragas_s11.py", score_file)
        return 0

    from scripts.eval_ragas_s11 import _evaluate, _print_table

    scores = _evaluate(rows)
    _print_table(rows, scores)
    return 0 if _gate_check(rows, scores) else 1


async def _run_monitor(config_name: str) -> int:
    rows = await collect_all(config_name)
    from scripts.eval_ragas_s11 import _evaluate

    scores = _evaluate(rows)
    for metric in MONITOR_METRICS:
        avg = _avg_metric(scores, metric)
        if avg is not None:
            print(f"{metric}: {avg:.3f}")
    return 0


async def _run_compare() -> int:
    results: dict[str, list[dict]] = {}
    for name in NAMED_CONFIGS:
        results[name] = await collect_all(name)
    _print_scoreboard(results)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Session 11 generation eval harness.")
    parser.add_argument("--gate", action="store_true", help="Gate against baseline")
    parser.add_argument("--monitor", action="store_true", help="Reference-free monitor")
    parser.add_argument("--compare", action="store_true", help="Run all named configs")
    parser.add_argument("--config", default="full", choices=sorted(NAMED_CONFIGS))
    parser.add_argument("--collect-only", type=Path, default=None, metavar="FILE")
    parser.add_argument("--score-file", type=Path, default=None, metavar="FILE")
    args = parser.parse_args()

    if args.score_file:
        from scripts.score_ragas_s11 import score_rows
        import os

        rows = json.loads(args.score_file.read_text(encoding="utf-8"))
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY required", file=sys.stderr)
            raise SystemExit(1)
        settings = get_settings()
        scores = score_rows(
            rows, openai_api_key=api_key, embedding_model=settings.EMBEDDING_MODEL
        )
        from scripts.eval_ragas_s11 import _print_table

        _print_table(rows, scores)
        raise SystemExit(0 if _gate_check(rows, scores) else 1)

    if args.compare:
        raise SystemExit(asyncio.run(_run_compare()))

    if args.monitor:
        raise SystemExit(asyncio.run(_run_monitor(args.config)))

    if args.gate or args.collect_only:
        raise SystemExit(asyncio.run(_run_gate(args.config, args.collect_only)))

    parser.print_help()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
