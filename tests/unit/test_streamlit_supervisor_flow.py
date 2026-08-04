"""Unit tests for Session 14 Streamlit supervisor helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from streamlit_ui.supervisor_flow import (
    load_sample_transcript,
    status_badge_label,
    supervisor_resume,
    supervisor_start,
    supervisor_state,
)


def test_status_badge_labels():
    assert "Awaiting" in status_badge_label("awaiting_human_review")
    assert "Validated" in status_badge_label("validated")


def test_load_sample_transcripts():
    happy = load_sample_transcript("happy_path")
    edge = load_sample_transcript("edge_case")
    assert "proveedores" in happy.lower() or "Portal" in happy
    assert "QKD" in edge or "cuántica" in edge


def test_supervisor_start_posts(monkeypatch):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "estimation_id": "e1",
        "state": "completed",
        "status": "validated",
    }
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.post.return_value = mock_response

    with patch("streamlit_ui.supervisor_flow.httpx.Client", return_value=mock_client):
        result = supervisor_start(
            "x" * 120,
            estimation_id="e1",
            api_root="http://test",
            api_key="k",
        )
    assert result["status"] == "validated"
    mock_client.post.assert_called_once()


def test_supervisor_resume_and_state():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"estimation_id": "e1", "state": "completed"}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.post.return_value = mock_response
    mock_client.get.return_value = mock_response

    with patch("streamlit_ui.supervisor_flow.httpx.Client", return_value=mock_client):
        supervisor_resume(
            "e1",
            decision="approve",
            api_root="http://test",
            api_key="k",
        )
        supervisor_state("e1", api_root="http://test", api_key="k")
    assert mock_client.post.called
    assert mock_client.get.called
