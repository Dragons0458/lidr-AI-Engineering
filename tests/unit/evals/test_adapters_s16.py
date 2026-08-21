"""Adapters keep RAG and graph payloads in engineer-days with explicit vs proxy abstention."""

from __future__ import annotations

from evals.production.adapters import normalize_graph_state, normalize_rag_response


def test_rag_days_and_explicit_abstention() -> None:
    outcome = normalize_rag_response(
        {
            "total_engineer_days": 40,
            "confidence": "high",
            "modules": [
                {
                    "name": "Auth",
                    "tasks": [
                        {
                            "name": "OAuth",
                            "sources": [{"document_id": "S07-FIN-001", "chunk_id": 1}],
                        }
                    ],
                }
            ],
            "sources": [{"source_id": 99, "relevance": "primary", "used_for": "auth"}],
            "assumptions": [{"description": "x", "impact": "low", "rationale": "y"}],
        },
        http_status=200,
        latency_ms=1200,
    )
    assert outcome.engineer_days == 40
    assert outcome.abstained is False
    assert outcome.abstention_signal == "explicit"
    assert "S07-FIN-001" in outcome.source_ids
    assert 99 not in {int(s) for s in outcome.source_ids if s.isdigit()}
    assert outcome.assumptions_count == 1


def test_rag_insufficient_is_explicit_abstention() -> None:
    outcome = normalize_rag_response(
        {
            "total_engineer_days": None,
            "confidence": "insufficient",
            "modules": [],
            "sources": [],
            "assumptions": [],
            "insufficient_context_explanation": "no analog",
        }
    )
    assert outcome.abstained is True
    assert outcome.engineer_days is None
    assert outcome.abstention_signal == "explicit"


def test_rag_null_days_counts_as_abstention() -> None:
    outcome = normalize_rag_response({"total_engineer_days": None, "confidence": "low"})
    assert outcome.abstained is True


def test_rag_transport_error() -> None:
    outcome = normalize_rag_response(None, http_status=502, error="HTTP 502")
    assert outcome.error == "HTTP 502"
    assert outcome.http_status == 502


def test_graph_days_from_estimate_not_hours() -> None:
    outcome = normalize_graph_state(
        {
            "estimate": {
                "total_engineer_hours": 280,
                "total_engineer_days": 35,
                "grounded_task_ratio": 1.0,
                "confidence": "high",
                "modules": [
                    {
                        "name": "Auth",
                        "tasks": [
                            {"name": "OAuth", "has_match": True, "estimated_hours": 120}
                        ],
                    }
                ],
            },
            "task_hours": [
                {
                    "module": "Auth",
                    "task": "OAuth",
                    "has_match": True,
                    "neighbors": [{"budget_id": "S07-FIN-001", "estimated_hours": 120}],
                }
            ],
        },
        llm_calls=3,
    )
    assert outcome.engineer_days == 35
    assert outcome.engineer_days != 280
    assert outcome.abstained is False
    assert outcome.abstention_signal == "proxy"
    assert outcome.source_ids == ["S07-FIN-001"]
    assert outcome.grounded_ratio == 1.0
    assert outcome.llm_calls == 3


def test_graph_proxy_abstention_zero_grounding() -> None:
    outcome = normalize_graph_state(
        {
            "estimate": {
                "total_engineer_days": 12,
                "grounded_task_ratio": 0.0,
                "confidence": "low",
                "modules": [
                    {"name": "X", "tasks": [{"name": "Y", "has_match": False}]}
                ],
            }
        }
    )
    assert outcome.abstained is True
    assert outcome.abstention_signal == "proxy"
    assert outcome.engineer_days is None


def test_graph_proxy_abstention_low_and_no_matches() -> None:
    outcome = normalize_graph_state(
        {
            "estimate": {
                "total_engineer_days": 8,
                "grounded_task_ratio": 0.4,
                "confidence": "low",
                "modules": [
                    {"name": "X", "tasks": [{"name": "Y", "has_match": False}]}
                ],
            }
        }
    )
    assert outcome.abstained is True
