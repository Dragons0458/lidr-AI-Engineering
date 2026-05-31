from collections.abc import Iterator

from fastapi.testclient import TestClient

API_PREFIX = "/api/v1"


def test_estimate_stream_emits_token_and_done(client: TestClient, monkeypatch) -> None:
    def fake_stream(*args, **kwargs) -> Iterator[str]:  # noqa: ARG001
        yield "Hello"
        yield "\nworld"

    monkeypatch.setattr(
        "app.routers.estimations.generate_estimation_stream", fake_stream
    )

    response = client.post(
        f"{API_PREFIX}/estimate/stream",
        json={
            "description": "Portal web con autenticacion y reportes detallados.",
            "project_type": "web_saas",
            "detail_level": "medium",
            "output_format": "phases_table",
            "evaluate": False,
        },
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    body = response.text
    assert "event: token" in body
    assert "data: Hello" in body
    assert "event: done" in body


def test_estimate_stream_rejects_short_description(client: TestClient) -> None:
    response = client.post(
        f"{API_PREFIX}/estimate/stream",
        json={
            "description": "short",
            "project_type": "web_saas",
            "detail_level": "medium",
            "output_format": "phases_table",
        },
    )
    assert response.status_code == 422


def test_estimate_stream_multiline_chunk(client: TestClient, monkeypatch) -> None:
    def fake_stream(*args, **kwargs) -> Iterator[str]:  # noqa: ARG001
        yield "line1\nline2"

    monkeypatch.setattr(
        "app.routers.estimations.generate_estimation_stream", fake_stream
    )

    response = client.post(
        f"{API_PREFIX}/estimate/stream",
        json={
            "description": "Portal web con autenticacion y reportes detallados.",
            "project_type": "web_saas",
            "detail_level": "medium",
            "output_format": "phases_table",
            "evaluate": False,
        },
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    assert response.text.count("data:") >= 2
