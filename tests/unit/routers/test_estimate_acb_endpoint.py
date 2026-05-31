from fastapi.testclient import TestClient

from app.main import app
from app.schemas.acb import ACBResponse, BossTrace
from app.schemas.estimation import EstimationResult, Phase, TokenUsage
from app.services.sessions import Session

API_PREFIX = "/api/v1"


def _acb_response() -> ACBResponse:
    result = EstimationResult(
        summary="Portal SaaS con autenticación y reportes para usuarios internos.",
        confidence_pct=85,
        phases=[
            Phase(
                name="Backend",
                base_hours=40,
                buffer_hours=5,
                team="2 devs",
                summary="API REST con autenticación y pruebas básicas del módulo.",
            )
        ],
        total_base_hours=40,
        total_buffer_hours=5,
        total_hours=45,
        total_cost_eur=3000,
    )
    return ACBResponse(
        result=result,
        model="test-model",
        provider="test",
        prompt_version="v3",
        latency_ms=100,
        usage=TokenUsage(input_tokens=10, output_tokens=20, tokens_used=30),
        cost_usd=0.01,
        acb=BossTrace(
            iterations=[],
            final_decision="accept",
            iterations_run=1,
        ),
    )


def test_acb_returns_structured_response(monkeypatch) -> None:
    Session.clear_all()
    session = Session.get_or_create("session-acb")

    def fake_generate_acb(request, *, session, tier_override=None, prompt_version="v3"):
        return _acb_response()

    def fake_extract(previous_metadata, request, llm_response):
        return previous_metadata

    monkeypatch.setattr(
        "app.routers.sessions.generate_estimation_acb", fake_generate_acb
    )
    monkeypatch.setattr("app.routers.sessions.extract_project_metadata", fake_extract)

    client = TestClient(app)
    response = client.post(
        f"{API_PREFIX}/sessions/{session.session_id}/estimate-acb",
        data={"description": "Project description with enough detail."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["acb"]["final_decision"] == "accept"
    assert body["result"]["total_hours"] == 45


def test_acb_unknown_session_returns_404() -> None:
    Session.clear_all()
    client = TestClient(app)

    response = client.post(
        f"{API_PREFIX}/sessions/missing/estimate-acb",
        data={"description": "Project description with enough detail."},
    )

    assert response.status_code == 404
