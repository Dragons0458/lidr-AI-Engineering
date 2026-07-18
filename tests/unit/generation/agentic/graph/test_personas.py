"""Matrix personas for graph agents."""

from __future__ import annotations

from app.generation.agentic.graph.personas import NODE_PERSONAS, persona_for


def test_persona_disabled_returns_none():
    assert persona_for("classifier_agent", enabled=False) is None


def test_unknown_node_returns_none():
    assert persona_for("mystery_node", enabled=True) is None


def test_all_llm_nodes_have_persona():
    for node in (
        "classifier_agent",
        "structure_agent",
        "recover_and_handover",
        "analysis_agent",
        "proposal_agent",
    ):
        assert node in NODE_PERSONAS


def test_persona_includes_guardrail():
    persona = persona_for("classifier_agent", enabled=True)
    assert persona is not None
    assert "never sacrifice correctness" in persona.lower()
