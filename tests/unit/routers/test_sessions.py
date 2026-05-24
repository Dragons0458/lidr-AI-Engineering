import json
from datetime import datetime
from io import BytesIO
from types import SimpleNamespace
from uuid import UUID

from docx import Document
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.estimation import EstimationResponse, TokenUsage
from app.services.sessions import ProjectMetadata, Session

API_PREFIX = "/api/v1"


def test_create_session_returns_uuid_and_registers_session() -> None:
    Session.clear_all()
    client = TestClient(app)

    response = client.post(f"{API_PREFIX}/sessions")

    assert response.status_code == 200
    session_id = response.json()["session_id"]
    assert UUID(session_id).version == 4
    assert Session.get(session_id) is not None


def test_get_session_debug_returns_snapshot() -> None:
    Session.clear_all()
    session = Session.get_or_create("session-123")
    session.metadata.project_name = "Nimbus"
    session.last_turn_observation = {"turn_index": 1, "latency_ms": 123}
    session.history.add_message("user", "Project description with enough detail.")
    session.history.add_message("assistant", "Estimated work breakdown")
    client = TestClient(app)

    response = client.get(f"{API_PREFIX}/sessions/{session.session_id}")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "session-123",
        "message_count": 2,
        "anchors_count": 0,
        "summary_chars": 0,
        "last_resolved_tier": None,
        "last_tier_rule": None,
        "summary": "",
        "anchors": [],
        "metadata": {
            "project_name": "Nimbus",
            "assumed_team_size": None,
            "mentioned_technologies": [],
            "excluded_technologies": [],
            "agreed_scope": None,
        },
        "last_turn_observation": {"turn_index": 1, "latency_ms": 123},
    }


def test_get_session_debug_returns_404_for_unknown_session() -> None:
    Session.clear_all()
    client = TestClient(app)

    response = client.get(f"{API_PREFIX}/sessions/missing")

    assert response.status_code == 404


def test_session_estimate_extracts_attachment_text(monkeypatch) -> None:
    Session.clear_all()
    session = Session.get_or_create("session-123")
    client = TestClient(app)
    captured = {}

    def fake_generate_estimation(
        request, prompt_version="v1", project_metadata=None, messages=None
    ):
        captured["request"] = request
        captured["prompt_version"] = prompt_version
        captured["project_metadata"] = project_metadata
        captured["messages"] = messages
        return SimpleNamespace()

    def fake_format_response(response, prompt_version="v1"):
        return EstimationResponse(
            estimation="Estimated work breakdown",
            model="test-model",
            provider="test-provider",
            timestamp=datetime(2026, 5, 18),
            usage=TokenUsage(tokens_used=10, cost_estimate=0.01),
            prompt_version=prompt_version,
        )

    def fake_extract_project_metadata(previous_metadata, request, llm_response):
        captured["previous_metadata"] = previous_metadata
        captured["metadata_request"] = request
        captured["llm_response"] = llm_response
        return ProjectMetadata(
            project_name="Estimated Project",
            mentioned_technologies=["FastAPI"],
        )

    monkeypatch.setattr(
        "app.routers.sessions.generate_estimation", fake_generate_estimation
    )
    monkeypatch.setattr("app.routers.sessions.format_response", fake_format_response)
    monkeypatch.setattr(
        "app.routers.sessions.extract_project_metadata",
        fake_extract_project_metadata,
    )

    response = client.post(
        f"{API_PREFIX}/sessions/{session.session_id}/estimate",
        data={"description": "Project description with enough detail."},
        files=[
            (
                "attachments",
                (
                    "scope.docx",
                    _build_docx_bytes("Authentication and reporting requirements."),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
            )
        ],
    )

    assert response.status_code == 200
    assert captured["prompt_version"] == "v1"
    assert captured["project_metadata"] == ProjectMetadata()
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][-1]["role"] == "user"
    assert captured["request"].description == "Project description with enough detail."
    assert captured["request"].attachments[0].filename == "scope.docx"
    assert (
        captured["request"].attachments[0].content
        == "Authentication and reporting requirements."
    )
    assert session.history.turns[-1].user.content == captured["request"].description
    assert session.history.turns[-1].assistant.content == "Estimated work breakdown"
    assert captured["previous_metadata"] == ProjectMetadata()
    assert captured["metadata_request"] == captured["request"]
    assert captured["llm_response"] == "Estimated work breakdown"
    assert session.metadata.project_name == "Estimated Project"
    assert session.metadata.mentioned_technologies == ["FastAPI"]


def test_session_estimate_accepts_attachments_without_description(monkeypatch) -> None:
    Session.clear_all()
    session = Session.get_or_create("session-123")
    client = TestClient(app)
    captured = {}

    def fake_generate_estimation(
        request, prompt_version="v1", project_metadata=None, messages=None
    ):
        captured["request"] = request
        return SimpleNamespace()

    def fake_format_response(response, prompt_version="v1"):
        return EstimationResponse(
            estimation="Estimated work breakdown",
            model="test-model",
            provider="test-provider",
            timestamp=datetime(2026, 5, 18),
            usage=TokenUsage(tokens_used=10, cost_estimate=0.01),
            prompt_version=prompt_version,
        )

    def fake_extract_project_metadata(previous_metadata, request, llm_response):
        return previous_metadata

    monkeypatch.setattr(
        "app.routers.sessions.generate_estimation", fake_generate_estimation
    )
    monkeypatch.setattr("app.routers.sessions.format_response", fake_format_response)
    monkeypatch.setattr(
        "app.routers.sessions.extract_project_metadata",
        fake_extract_project_metadata,
    )

    response = client.post(
        f"{API_PREFIX}/sessions/{session.session_id}/estimate",
        data={"description": ""},
        files=[
            (
                "attachments",
                (
                    "scope.docx",
                    _build_docx_bytes("Authentication and reporting requirements."),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
            )
        ],
    )

    assert response.status_code == 200
    assert (
        captured["request"].description
        == "Estimar el proyecto con base en los adjuntos proporcionados."
    )
    assert captured["request"].attachments[0].content == (
        "Authentication and reporting requirements."
    )


def test_session_estimate_emits_turn_observed(monkeypatch) -> None:
    Session.clear_all()
    session = Session.get_or_create("session-123")
    client = TestClient(app)
    log_events = []

    def fake_generate_estimation(
        request, prompt_version="v1", project_metadata=None, messages=None
    ):
        return SimpleNamespace()

    def fake_format_response(response, prompt_version="v1"):
        return EstimationResponse(
            estimation="Estimated work breakdown",
            model="test-model",
            provider="test-provider",
            timestamp=datetime(2026, 5, 18),
            usage=TokenUsage(
                tokens_used=30,
                input_tokens=20,
                output_tokens=10,
                cost_estimate=0.02,
            ),
            prompt_version=prompt_version,
            latency_ms=123,
        )

    def fake_extract_project_metadata(previous_metadata, request, llm_response):
        return previous_metadata

    def fake_log_info(event, **kwargs):
        log_events.append((event, kwargs))

    monkeypatch.setattr(
        "app.routers.sessions.generate_estimation", fake_generate_estimation
    )
    monkeypatch.setattr("app.routers.sessions.format_response", fake_format_response)
    monkeypatch.setattr(
        "app.routers.sessions.extract_project_metadata",
        fake_extract_project_metadata,
    )
    monkeypatch.setattr("app.routers.sessions.log.info", fake_log_info)

    response = client.post(
        f"{API_PREFIX}/sessions/{session.session_id}/estimate",
        data={"description": "Project description with enough detail."},
    )

    assert response.status_code == 200
    assert log_events == [
        (
            "turn_observed",
            {
                "turn_index": 1,
                "session_id": "session-123",
                "enriched_transcript_chars": 39,
                "attachments_total_chars": 0,
                "messages_in_window": 2,
                "anchors_count": 0,
                "summary_chars": 0,
                "tokens_in": 20,
                "tokens_out": 10,
                "cost_usd": 0.02,
                "latency_ms": 123,
                "cache_hit_kind": "none",
                "last_resolved_tier": None,
            },
        )
    ]
    assert session.last_turn_observation == log_events[0][1]


def test_session_estimate_rejects_empty_description_without_attachments(
    monkeypatch,
) -> None:
    Session.clear_all()
    session = Session.get_or_create("session-123")
    client = TestClient(app)

    def fail_generate_estimation(request, prompt_version="v1"):
        raise AssertionError("LLM call should not run without usable input")

    monkeypatch.setattr(
        "app.routers.sessions.generate_estimation", fail_generate_estimation
    )

    response = client.post(
        f"{API_PREFIX}/sessions/{session.session_id}/estimate",
        data={"description": ""},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "description is required when no attachments are provided."
    )


def test_session_estimate_accepts_form_controls_and_reference_projects(
    monkeypatch,
) -> None:
    Session.clear_all()
    session = Session.get_or_create("session-123")
    client = TestClient(app)
    captured = {}

    def fake_generate_estimation(
        request, prompt_version="v1", project_metadata=None, messages=None
    ):
        captured["request"] = request
        captured["prompt_version"] = prompt_version
        return SimpleNamespace()

    def fake_format_response(response, prompt_version="v1"):
        return EstimationResponse(
            estimation="Estimated work breakdown",
            model="test-model",
            provider="test-provider",
            timestamp=datetime(2026, 5, 18),
            usage=TokenUsage(tokens_used=10, cost_estimate=0.01),
            prompt_version=prompt_version,
        )

    def fake_extract_project_metadata(previous_metadata, request, llm_response):
        return previous_metadata

    monkeypatch.setattr(
        "app.routers.sessions.generate_estimation", fake_generate_estimation
    )
    monkeypatch.setattr("app.routers.sessions.format_response", fake_format_response)
    monkeypatch.setattr(
        "app.routers.sessions.extract_project_metadata",
        fake_extract_project_metadata,
    )

    response = client.post(
        f"{API_PREFIX}/sessions/{session.session_id}/estimate",
        data={
            "description": "Project description with enough detail.",
            "project_type": "mobile_app",
            "detail_level": "detailed",
            "output_format": "narrative",
            "reference_projects": json.dumps(
                [
                    {
                        "name": "Portal interno",
                        "summary": "Portal con autenticacion, reportes y panel admin.",
                        "estimated_hours": 240,
                        "team": "2 backend, 1 frontend",
                        "outcome": "Entregado en seis semanas",
                    }
                ]
            ),
        },
        params={"prompt_version": "v2"},
    )

    assert response.status_code == 200
    assert captured["prompt_version"] == "v2"
    assert captured["request"].project_type.value == "mobile_app"
    assert captured["request"].detail_level.value == "detailed"
    assert captured["request"].output_format.value == "narrative"
    assert captured["request"].reference_projects[0].name == "Portal interno"
    assert captured["request"].reference_projects[0].estimated_hours == 240


def test_session_estimate_rejects_unsupported_attachment_type(monkeypatch) -> None:
    Session.clear_all()
    session = Session.get_or_create("session-123")
    client = TestClient(app)

    def fail_generate_estimation(request, prompt_version="v1"):
        raise AssertionError("LLM call should not run for unsupported attachments")

    monkeypatch.setattr(
        "app.routers.sessions.generate_estimation", fail_generate_estimation
    )

    response = client.post(
        f"{API_PREFIX}/sessions/{session.session_id}/estimate",
        data={"description": "Project description with enough detail."},
        files=[
            ("attachments", ("scope.txt", b"Plain text is not allowed.", "text/plain"))
        ],
    )

    assert response.status_code == 415
    assert "Only PDF and DOCX files are allowed" in response.json()["detail"]


def test_session_estimate_returns_404_for_unknown_session() -> None:
    Session.clear_all()
    client = TestClient(app)

    response = client.post(
        f"{API_PREFIX}/sessions/missing/estimate",
        data={"description": "Project description with enough detail."},
    )

    assert response.status_code == 404


def _build_docx_bytes(text: str) -> bytes:
    buffer = BytesIO()
    document = Document()
    document.add_paragraph(text)
    document.save(buffer)
    return buffer.getvalue()
