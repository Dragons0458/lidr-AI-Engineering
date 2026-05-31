from app.services.estimation_service import EstimationGenerationResult

API_PREFIX = "/api/v1"


def test_estimate_endpoint_returns_validation_latency_and_finish_reason(
    client, monkeypatch
) -> None:
    estimation = """## Estimacion: Portal

| Phase | Tasks | Hours | Team |
|---|---|---:|---|
| Discovery | Scope | 10 | PM |
| Backend | API | 40 | Backend Engineer |
| Total | - | 50 | Core team |

Total estimated hours: 50h
Equipo recomendado: PM y Backend Engineer
Duracion estimada: 4 semanas
"""

    def fake_generate_estimation(request, prompt_version="v1"):
        return EstimationGenerationResult(
            estimation=estimation,
            model="test-model",
            provider="openai",
            latency_ms=123,
            input_tokens=10,
            output_tokens=12,
            finish_reason="stop",
        )

    monkeypatch.setattr(
        "app.routers.estimations.generate_estimation", fake_generate_estimation
    )

    response = client.post(
        f"{API_PREFIX}/estimate",
        json={
            "description": "Portal web con autenticacion y reportes.",
            "project_type": "web_saas",
            "detail_level": "medium",
            "output_format": "phases_table",
            "evaluate": True,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["latency_ms"] == 123
    assert payload["finish_reason"] == "stop"
    assert payload["usage"]["input_tokens"] == 10
    assert payload["usage"]["output_tokens"] == 12
    assert payload["validation"]["score"] == 1.0


def test_estimate_endpoint_can_skip_validation(client, monkeypatch) -> None:
    def fake_generate_estimation(request, prompt_version="v1"):
        return EstimationGenerationResult(
            estimation="## Estimacion\nTotal estimated hours: 20h",
            model="test-model",
            provider="openai",
            latency_ms=50,
            input_tokens=10,
            output_tokens=12,
            finish_reason="stop",
        )

    monkeypatch.setattr(
        "app.routers.estimations.generate_estimation", fake_generate_estimation
    )

    response = client.post(
        f"{API_PREFIX}/estimate",
        json={
            "description": "Portal web con autenticacion y reportes.",
            "project_type": "web_saas",
            "detail_level": "medium",
            "output_format": "line_items",
            "evaluate": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["validation"] is None
