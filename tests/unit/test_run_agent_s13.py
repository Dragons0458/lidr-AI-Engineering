"""Unit tests for scripts/run_agent_s13.py (no network)."""

from __future__ import annotations

from scripts.run_agent_s13 import build_payload


def test_build_payload_uses_provided_estimation_id():
    payload = build_payload("hello transcript", "fixed-id")
    assert payload == {
        "estimation_id": "fixed-id",
        "transcript": "hello transcript",
    }


def test_build_payload_generates_estimation_id_when_omitted():
    payload = build_payload("hello", None)
    assert payload["transcript"] == "hello"
    assert payload["estimation_id"].startswith("s13-")
