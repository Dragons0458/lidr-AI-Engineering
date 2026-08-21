"""The Session 16 golden set is a real, auditable measuring stick."""

from __future__ import annotations

import json
from pathlib import Path

from evals.production.schemas import GoldenSet
from evals.production.validate import load_golden_set, validate_golden_set

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "evals" / "golden_set_s16.json"


def test_golden_set_file_validates() -> None:
    golden = load_golden_set(GOLDEN)
    assert isinstance(golden, GoldenSet)
    assert len(golden.cases) >= 5
    assert sum(1 for case in golden.cases if case.expect_abstention) >= 1
    assert golden.units == "engineer_days"
    assert golden.hours_per_day == 8


def test_golden_set_ids_unique_and_ranges_hold() -> None:
    golden = load_golden_set(GOLDEN)
    ids = [case.id for case in golden.cases]
    assert len(ids) == len(set(ids))
    for case in golden.cases:
        low, high = case.acceptable_range
        assert low <= case.expected_engineer_days <= high


def test_transcripts_exist_and_are_long_enough() -> None:
    golden = load_golden_set(GOLDEN)
    errors = validate_golden_set(golden, repo_root=ROOT)
    assert errors == []


def test_expected_sources_are_real_budget_ids() -> None:
    corpus = json.loads(
        (ROOT / "data" / "budgets_sample.json").read_text(encoding="utf-8")
    )
    known = {row["budget_id"] for row in corpus}
    golden = load_golden_set(GOLDEN)
    for case in golden.cases:
        for budget_id in case.expected_sources_include:
            assert budget_id in known
            assert budget_id.startswith("S07-")
