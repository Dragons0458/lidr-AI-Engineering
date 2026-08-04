"""Unit tests for Session 14 supervisor signal helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.generation.agentic.graph.supervisor_state import (
    apply_human_decision,
    build_state_digest,
    compute_confidence,
    detect_review_risks,
    estimate_for_historical_band,
    historical_band,
    is_outside_historical_band,
    requires_human_review,
)


def _estimate(total: float, *hours: float) -> dict:
    return {
        "components": [
            {
                "name": f"c{i}",
                "estimated_hours": h,
                "cited_chunk_ids": [],
                "rationale": "",
            }
            for i, h in enumerate(hours)
        ],
        "total_hours": total,
        "assumptions": [],
        "confidence": "low",
    }


def _match(component_id: str, amount: float, chunk_id: int = 1) -> dict:
    return {
        "component_id": component_id,
        "chunk_id": chunk_id,
        "reference_budget_id": "b1",
        "amount": amount,
        "distance": 0.1,
    }


def test_digest_excludes_transcript_and_is_bounded():
    state = {
        "transcript": "SECRET PII " * 500,
        "requirements": ["a", "b"],
        "components": [
            {"component_id": "c1", "name": "n", "category": "m", "description": ""}
        ],
        "budget_matches": [_match("c1", 10)],
        "estimate": _estimate(11.5, 11.5),
        "validation": {"ok": True, "issues": []},
        "confidence": 0.9,
        "routing_steps": 3,
        "search_completed": True,
    }
    digest = build_state_digest(state)
    assert "SECRET" not in digest
    assert "PII" not in digest
    assert "requirements=2" in digest
    assert len(digest) <= 500


def test_digest_uses_the_configured_grounding_distance():
    state = {
        "components": [{"component_id": "a", "name": "API", "category": "Backend"}],
        "budget_matches": [_match("a", 20)],
    }
    state["budget_matches"][0]["distance"] = 0.50

    assert "grounded=0/1" in build_state_digest(state, grounding_max_distance=0.45)
    assert "grounded=1/1" in build_state_digest(state, grounding_max_distance=0.51)


def test_compute_confidence_coverage_and_penalties():
    components = [
        {"component_id": "a", "name": "A", "category": "m", "description": ""},
        {"component_id": "b", "name": "B", "category": "m", "description": ""},
    ]
    full = {
        "components": components,
        "budget_matches": [_match("a", 10), _match("b", 20, chunk_id=2)],
        "estimate": _estimate(34.5, 11.5, 23.0),
        "validation": {"ok": True, "issues": []},
    }
    assert compute_confidence(full) == 1.0

    half = {
        "components": components,
        "budget_matches": [_match("a", 10)],
        "estimate": _estimate(11.5, 11.5, 0.0),
        "validation": {"ok": True, "issues": []},
    }
    assert compute_confidence(half) == 0.5

    issues = {
        **full,
        "validation": {"ok": False, "issues": ["x", "y"]},
    }
    assert compute_confidence(issues) == pytest.approx(0.8)


def test_historical_band_and_out_of_range():
    matches = [_match("a", 10), _match("a", 30, chunk_id=2), _match("b", 20)]
    band = historical_band(matches, factor=2.0)
    # medians: a=20, b=20 → centre=40 → [20, 80]
    assert band["centre"] == 40.0
    assert band["low"] == 20.0
    assert band["high"] == 80.0

    assert not is_outside_historical_band(_estimate(40, 20, 20), matches, factor=2.0)
    assert is_outside_historical_band(_estimate(10, 5, 5), matches, factor=2.0)
    assert is_outside_historical_band(_estimate(200, 100, 100), matches, factor=2.0)


def test_historical_band_compares_only_the_grounded_component_subset():
    state = {
        "estimate": _estimate(165, 54, 36, 42, 33),
        "component_anchors": [
            {"component_id": "auth", "estimated_hours": 54},
            {"component_id": "roles", "estimated_hours": 36},
            {"component_id": "api", "estimated_hours": 42},
            {"component_id": "sap", "estimated_hours": 33},
        ],
    }
    grounded = [
        _match("roles", 30),
        _match("roles", 36),
        _match("api", 39),
    ]

    comparable = estimate_for_historical_band(state, grounded)

    assert comparable == {"total_hours": 78.0}
    assert not is_outside_historical_band(comparable, grounded, factor=2.0)


def test_requires_human_review_signals_separately_and_combined():
    base = {
        "components": [
            {"component_id": "a", "name": "A", "category": "m", "description": ""}
        ],
        "budget_matches": [_match("a", 10)],
        "estimate": _estimate(11.5, 11.5),
        "confidence": 0.9,
        "validation": {"ok": True, "issues": []},
    }
    needs, reasons = requires_human_review(base, confidence_threshold=0.7)
    assert needs is False
    assert reasons == []

    low = {**base, "confidence": 0.2}
    needs, reasons = requires_human_review(low, confidence_threshold=0.7)
    assert needs is True
    assert "low_confidence" in reasons

    none = {**base, "budget_matches": [], "confidence": 0.9}
    needs, reasons = requires_human_review(none, confidence_threshold=0.7)
    assert "no_precedent" in reasons

    oom = {
        **base,
        "estimate": _estimate(1000, 1000),
        "budget_matches": [_match("a", 10)],
        "confidence": 0.9,
    }
    needs, reasons = requires_human_review(
        oom, confidence_threshold=0.7, out_of_range_factor=2.0
    )
    assert "out_of_range" in reasons


def test_far_neighbours_do_not_count_as_precedent():
    """Distant false neighbours must not fake grounding / skip HITL."""
    far = _match("a", 10)
    far["distance"] = 0.55  # inside recall band, outside grounding band
    state = {
        "components": [
            {
                "component_id": "a",
                "name": "REST API",
                "category": "m",
                "description": "",
            }
        ],
        "budget_matches": [far],
        "estimate": _estimate(11.5, 11.5),
        "confidence": 1.0,
        "validation": {"ok": True, "issues": []},
    }
    assert compute_confidence(state) == 0.0
    needs, reasons = requires_human_review(state, confidence_threshold=0.7)
    assert needs is True
    assert "no_precedent" in reasons

    near = _match("a", 10)
    near["distance"] = 0.40
    ok = {**state, "budget_matches": [near], "confidence": 0.9}
    assert compute_confidence(ok) == 1.0
    needs, reasons = requires_human_review(ok, confidence_threshold=0.7)
    assert needs is False
    assert reasons == []


def test_real_samples_separate_ordinary_scope_from_high_risk_scope():
    happy = Path("exercises/session-14/sample_transcript_happy_path.txt").read_text(
        encoding="utf-8"
    )
    edge = Path("exercises/session-14/sample_transcript_edge_case.txt").read_text(
        encoding="utf-8"
    )

    assert detect_review_risks({"transcript": happy}) == []
    edge_flags = detect_review_risks({"transcript": edge})
    assert "novel_cryptography" in edge_flags
    assert "legacy_or_undocumented_integration" in edge_flags
    assert "custom_security_hardware" in edge_flags
    assert "open_scope" in edge_flags


def test_live_like_happy_distances_pass_while_same_grounding_with_risk_pauses():
    components = [
        {"component_id": f"c{i}", "name": f"ordinary-{i}", "category": "m"}
        for i in range(5)
    ]
    # Minima observed across live happy-path checkpoints. At 0.55, ordinary
    # auth/API/SAP components remain grounded while risk flags separate edge cases.
    distances = [0.5381, 0.4329, 0.5065, 0.5439, 0.5466]
    matches = [
        {**_match(f"c{i}", 30 + i), "distance": distance}
        for i, distance in enumerate(distances)
    ]
    state = {
        "components": components,
        "budget_matches": matches,
        "estimate": _estimate(172.5, 30, 30, 30, 40, 42.5),
        "validation": {"ok": True, "issues": []},
    }
    confidence = compute_confidence(state, grounding_max_distance=0.55)
    happy = {**state, "confidence": confidence}

    assert confidence == 1.0
    assert requires_human_review(
        happy,
        confidence_threshold=0.6,
        min_grounded_ratio=0.5,
        out_of_range_factor=2.0,
        grounding_max_distance=0.55,
    ) == (False, [])

    risky = {**happy, "transcript": "QKD over an undocumented COBOL mainframe"}
    needs, reasons = requires_human_review(
        risky,
        confidence_threshold=0.6,
        min_grounded_ratio=0.5,
        out_of_range_factor=2.0,
        grounding_max_distance=0.55,
    )
    assert needs is True
    assert "high_risk_scope" in reasons


def test_apply_human_decision_approve_adjust_reject():
    state = {"estimate": _estimate(20, 10, 10), "status": None}

    approved = apply_human_decision(state, {"action": "approve"})
    assert approved["status"] == "validated"
    assert approved["estimate"]["total_hours"] == 20

    rejected = apply_human_decision(state, {"action": "reject", "notes": "no"})
    assert rejected["status"] == "rejected"

    adjusted = apply_human_decision(
        state,
        {
            "action": "adjust",
            "estimate_overrides": {
                "components": [
                    {"name": "c0", "estimated_hours": 5},
                    {"name": "c1", "estimated_hours": 7},
                ]
            },
        },
    )
    assert adjusted["status"] == "needs_review"
    assert adjusted["estimate"]["total_hours"] == 12.0
