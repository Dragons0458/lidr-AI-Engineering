"""Unit tests for Session 11 estimate serialization helpers."""

from __future__ import annotations

from app.generation.rag.serialization import compact_response_for_relevancy


def test_compact_response_strips_chunk_citations_and_assumptions():
    answer = (
        "Confidence: high\n"
        "Reasoning: Grounded in ecommerce budgets.\n"
        "Total engineer-days: 40\n"
        "\n"
        "## Module: Payments\n"
        "- Checkout (18 engineer-days, grounded)\n"
        '  [chunk 12 / S07-ECO-001] "estimated hours: 140"\n'
        "\n"
        "Assumptions:\n"
        "- Mobile app scope inferred (high)\n"
    )
    compact = compact_response_for_relevancy(answer)
    assert "[chunk" not in compact
    assert "Assumptions" not in compact
    assert "Confidence:" not in compact
    assert "Reasoning:" not in compact
    assert "Checkout (18 engineer-days, grounded)" in compact
    assert "Total engineer-days: 40" in compact
