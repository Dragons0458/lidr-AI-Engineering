"""Load + filesystem checks for the Session 16 golden set (no network)."""

from __future__ import annotations

import json
from pathlib import Path

from evals.production.schemas import GoldenSet

MIN_TRANSCRIPT_CHARS = 100
_BUDGET_ID_PREFIX = "S07-"


def load_golden_set(path: Path) -> GoldenSet:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return GoldenSet.model_validate(payload)


def load_budget_ids(corpus_path: Path) -> set[str]:
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    else:
        rows = payload.get("budgets") or payload.get("items") or []
    return {
        str(row["budget_id"])
        for row in rows
        if isinstance(row, dict) and row.get("budget_id")
    }


def validate_golden_set(
    golden: GoldenSet,
    *,
    repo_root: Path,
    budget_ids: set[str] | None = None,
) -> list[str]:
    """Return human-readable errors. Empty list means the set is runnable."""
    errors: list[str] = []
    known = budget_ids
    if known is None:
        corpus = repo_root / golden.corpus
        if not corpus.is_file():
            errors.append(f"corpus not found: {golden.corpus}")
            known = set()
        else:
            known = load_budget_ids(corpus)

    for case in golden.cases:
        transcript = repo_root / case.transcript_path
        if not transcript.is_file():
            errors.append(f"{case.id}: transcript missing ({case.transcript_path})")
            continue
        text = transcript.read_text(encoding="utf-8").strip()
        if len(text) < MIN_TRANSCRIPT_CHARS:
            errors.append(
                f"{case.id}: transcript has {len(text)} chars "
                f"(need ≥{MIN_TRANSCRIPT_CHARS})"
            )
        for budget_id in case.expected_sources_include:
            if budget_id not in known:
                errors.append(
                    f"{case.id}: expected source {budget_id!r} is not in {golden.corpus}"
                )
            if not str(budget_id).startswith(_BUDGET_ID_PREFIX):
                errors.append(
                    f"{case.id}: expected source {budget_id!r} does not look like a budget_id"
                )
    return errors
