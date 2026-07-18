"""Unit tests for scripts/run_agent_s13.py (no network)."""

from __future__ import annotations

from unittest.mock import MagicMock


from scripts.run_agent_s13 import build_payload, canned_decision, run_flow


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


def test_canned_decisions_by_gate():
    assert canned_decision("structure_review", want_proposal=True) == {"approved": True}
    assert canned_decision("final_review", want_proposal=False) == {
        "validated": True,
        "want_proposal": False,
    }


def test_run_flow_resumes_until_completed():
    resume_count = 0

    def fake_post(url, **kwargs):
        nonlocal resume_count
        response = MagicMock()
        if url.endswith("/graph"):
            response.json.return_value = {
                "state": "paused",
                "pending_gate": {"gate": "structure_review"},
            }
        elif url.endswith("/resume"):
            resume_count += 1
            if resume_count == 1:
                response.json.return_value = {
                    "state": "paused",
                    "pending_gate": {"gate": "final_review"},
                }
            else:
                response.json.return_value = {
                    "state": "completed",
                    "status": "validated",
                }
        else:
            raise AssertionError(url)
        response.raise_for_status = MagicMock()
        return response

    def fake_get(url, **kwargs):
        response = MagicMock()
        response.json.return_value = {"state": "completed", "status": "validated"}
        response.raise_for_status = MagicMock()
        return response

    client = MagicMock()
    client.post.side_effect = fake_post
    client.get.side_effect = fake_get

    state = run_flow(
        client,
        base_url="http://localhost:8000",
        api_key="key",
        payload={"estimation_id": "e1", "transcript": "hello"},
        want_proposal=True,
    )
    assert state["status"] == "validated"
    assert resume_count == 2
