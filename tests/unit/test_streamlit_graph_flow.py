"""Unit tests for Streamlit graph flow helpers."""

from __future__ import annotations

from app.generation.agentic.graph.activity import _NODE_KEYS
from streamlit_ui.graph_flow import (
    GRAPH_NODES,
    activity_by_node,
    estimate_rows,
    proposal_pdf_bytes,
    rows_to_estimate_overrides,
)


def test_graph_nodes_catalog_matches_activity_keys():
    catalog_keys = {node["key"] for node in GRAPH_NODES}
    activity_keys = set(_NODE_KEYS.values())
    assert catalog_keys == activity_keys


def test_estimate_rows_and_overrides_roundtrip():
    estimate = {
        "modules": [
            {
                "name": "Backend",
                "tasks": [
                    {
                        "name": "API",
                        "description": "REST",
                        "estimated_hours": 40.0,
                        "reliability": 0.9,
                        "has_match": True,
                    }
                ],
            }
        ]
    }
    rows = estimate_rows(estimate)
    overrides = rows_to_estimate_overrides(rows)
    assert overrides["modules"][0]["tasks"][0]["estimated_hours"] == 40.0


def test_activity_by_node_groups_messages():
    grouped = activity_by_node(
        [
            {"node": "classifier", "message": "Complejidad: high"},
            {"node": "classifier", "message": "done"},
        ]
    )
    assert grouped["classifier"] == ["Complejidad: high", "done"]


def test_proposal_pdf_bytes_not_empty():
    payload = proposal_pdf_bytes("Demo", "Proposal body text for the client.")
    assert isinstance(payload, bytes)
    assert len(payload) > 100
