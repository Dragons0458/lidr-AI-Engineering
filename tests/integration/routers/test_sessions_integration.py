from datetime import datetime
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import get_settings
from app.main import app
from app.schemas.estimation import EstimationResponse, TokenUsage
from app.services.sessions import ProjectMetadata, Session

API_PREFIX = "/api/v1"


@pytest.mark.anyio
async def test_session_links_requests_and_updates_project_metadata(
    monkeypatch,
) -> None:
    Session.clear_all()
    captured_calls = []

    def fake_generate_estimation(
        request, prompt_version="v1", project_metadata=None, messages=None
    ):
        captured_calls.append(
            {
                "request": request,
                "project_metadata": project_metadata,
                "messages": messages,
            }
        )
        return SimpleNamespace(content=f"estimate for {request.description}")

    def fake_format_response(response, prompt_version="v1"):
        return _estimation_response(response.content, prompt_version=prompt_version)

    def fake_extract_project_metadata(previous_metadata, request, llm_response):
        if "CRM interno" in request.description:
            return ProjectMetadata(
                project_name="CRM interno",
                mentioned_technologies=["FastAPI"],
            )

        assert previous_metadata.project_name == "CRM interno"
        return ProjectMetadata(
            project_name=previous_metadata.project_name,
            assumed_team_size=4,
            mentioned_technologies=[
                *previous_metadata.mentioned_technologies,
                "React",
            ],
            agreed_scope="Modulo de reportes y dashboard ejecutivo.",
        )

    monkeypatch.setattr(
        "app.routers.sessions.generate_estimation", fake_generate_estimation
    )
    monkeypatch.setattr("app.routers.sessions.format_response", fake_format_response)
    monkeypatch.setattr(
        "app.routers.sessions.extract_project_metadata",
        fake_extract_project_metadata,
    )

    async with _async_client() as client:
        create_response = await client.post(f"{API_PREFIX}/sessions")
        session_id = create_response.json()["session_id"]

        first_response = await client.post(
            f"{API_PREFIX}/sessions/{session_id}/estimate",
            data={"description": "Estimar CRM interno con API en FastAPI."},
        )
        second_response = await client.post(
            f"{API_PREFIX}/sessions/{session_id}/estimate",
            data={
                "description": (
                    "Agregar dashboard ejecutivo en React para el mismo CRM."
                )
            },
        )

    assert first_response.status_code == 200
    assert first_response.json()["project_metadata"] == {
        "project_name": "CRM interno",
        "assumed_team_size": None,
        "mentioned_technologies": ["FastAPI"],
        "excluded_technologies": [],
        "agreed_scope": None,
    }
    assert second_response.status_code == 200
    assert second_response.json()["project_metadata"] == {
        "project_name": "CRM interno",
        "assumed_team_size": 4,
        "mentioned_technologies": ["FastAPI", "React"],
        "excluded_technologies": [],
        "agreed_scope": "Modulo de reportes y dashboard ejecutivo.",
    }

    assert captured_calls[0]["project_metadata"] == ProjectMetadata()
    assert captured_calls[1]["project_metadata"].project_name == "CRM interno"
    assert captured_calls[1]["project_metadata"].mentioned_technologies == ["FastAPI"]
    assert "<project_name>CRM interno</project_name>" in (
        captured_calls[1]["messages"][0]["content"]
    )
    assert Session.get(session_id).metadata.assumed_team_size == 4


@pytest.mark.anyio
async def test_pdf_attachment_content_changes_estimation_output(monkeypatch) -> None:
    Session.clear_all()

    def fake_generate_estimation(
        request, prompt_version="v1", project_metadata=None, messages=None
    ):
        attachment_text = " ".join(
            attachment.content for attachment in request.attachments or []
        )
        if "SSO audit trail" in attachment_text:
            return SimpleNamespace(content="Estimate includes SSO audit trail work")

        return SimpleNamespace(content="Baseline estimate without compliance add-on")

    def fake_format_response(response, prompt_version="v1"):
        return _estimation_response(response.content, prompt_version=prompt_version)

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

    async with _async_client() as client:
        baseline_session_id = (
            await client.post(f"{API_PREFIX}/sessions")
        ).json()["session_id"]
        pdf_session_id = (
            await client.post(f"{API_PREFIX}/sessions")
        ).json()["session_id"]

        baseline_response = await client.post(
            f"{API_PREFIX}/sessions/{baseline_session_id}/estimate",
            data={"description": "Build a customer portal with backend and frontend."},
        )
        pdf_response = await client.post(
            f"{API_PREFIX}/sessions/{pdf_session_id}/estimate",
            data={"description": "Build a customer portal with backend and frontend."},
            files=[
                (
                    "attachments",
                    (
                        "requirements.pdf",
                        _build_pdf_bytes("SSO audit trail requirements"),
                        "application/pdf",
                    ),
                )
            ],
        )

    assert baseline_response.status_code == 200
    assert pdf_response.status_code == 200
    assert baseline_response.json()["estimation"] == (
        "Baseline estimate without compliance add-on"
    )
    assert pdf_response.json()["estimation"] == (
        "Estimate includes SSO audit trail work"
    )


@pytest.mark.anyio
async def test_effective_history_sent_to_llm_never_exceeds_configured_max_turns(
    monkeypatch,
) -> None:
    Session.clear_all()
    max_turns = get_settings().CONVERSATION_MAX_TURNS
    captured_messages = []

    def fake_generate_estimation(
        request, prompt_version="v1", project_metadata=None, messages=None
    ):
        captured_messages.append(messages)
        return SimpleNamespace(content=f"estimate for {request.description}")

    def fake_format_response(response, prompt_version="v1"):
        return _estimation_response(response.content, prompt_version=prompt_version)

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

    async with _async_client() as client:
        session_id = (await client.post(f"{API_PREFIX}/sessions")).json()["session_id"]

        for turn_number in range(8):
            response = await client.post(
                f"{API_PREFIX}/sessions/{session_id}/estimate",
                data={
                    "description": (
                        f"Turn {turn_number}: estimate reporting workflow updates."
                    )
                },
            )
            assert response.status_code == 200

    assert len(captured_messages) == 8
    for messages in captured_messages:
        historical_messages = messages[1:-1]
        assert len(historical_messages) <= max_turns * 2

    last_historical_messages = captured_messages[-1][1:-1]
    last_historical_user_messages = [
        message["content"]
        for message in last_historical_messages
        if message["role"] == "user"
    ]
    assert len(last_historical_user_messages) == max_turns
    assert last_historical_user_messages[0] == (
        "Turn 1: estimate reporting workflow updates."
    )
    assert last_historical_user_messages[-1] == (
        "Turn 6: estimate reporting workflow updates."
    )


def _async_client() -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    )


def _estimation_response(
    estimation: str,
    *,
    prompt_version: str = "v1",
) -> EstimationResponse:
    return EstimationResponse(
        estimation=estimation,
        model="test-model",
        provider="test-provider",
        timestamp=datetime(2026, 5, 18),
        usage=TokenUsage(tokens_used=10, cost_estimate=0.01),
        prompt_version=prompt_version,
    )


def _build_pdf_bytes(text: str) -> bytes:
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET"
    objects = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            "3 0 obj\n"
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\n"
            "endobj\n"
        ),
        "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        (
            f"5 0 obj\n<< /Length {len(stream.encode('latin-1'))} >>\n"
            f"stream\n{stream}\nendstream\nendobj\n"
        ),
    ]
    data = "%PDF-1.4\n"
    offsets = []
    for pdf_object in objects:
        offsets.append(len(data.encode("latin-1")))
        data += pdf_object

    xref_offset = len(data.encode("latin-1"))
    data += f"xref\n0 {len(objects) + 1}\n"
    data += "0000000000 65535 f \n"
    for offset in offsets:
        data += f"{offset:010d} 00000 n \n"

    data += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    return data.encode("latin-1")
