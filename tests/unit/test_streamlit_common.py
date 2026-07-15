import httpx

from streamlit_ui.common import (
    DEFAULT_AGENT_MODELS,
    agent_model_label,
    fetch_available_agent_models,
    format_api_error,
    format_guardrail_detail,
    get_api_root_url,
    parse_error_detail,
)
from streamlit_ui.rag import (
    STRATEGY_CATALOG,
    build_retrieval_update_payload,
    build_settings_update_payload,
    cost_hint,
    label_for,
)


def test_format_guardrail_detail() -> None:
    detail = {
        "reason": "prompt_injection",
        "message": "Potential prompt injection detected in input",
    }
    text = format_guardrail_detail(detail)
    assert text is not None
    assert "Manipulación de instrucciones" in text
    assert "ignore previous instructions" in text or "Potential prompt" in text


def test_format_api_error_guardrail_400() -> None:
    request = httpx.Request("POST", "http://localhost:8000/api/v1/estimate")
    response = httpx.Response(
        400,
        request=request,
        json={
            "detail": {
                "reason": "pii",
                "message": "Email address detected in input",
            }
        },
    )
    exc = httpx.HTTPStatusError("bad", request=request, response=response)
    message = format_api_error(exc, api_base_url="http://localhost:8000/api/v1")

    assert "400" in message
    assert "Datos personales" in message
    assert "Email address" in message


def test_parse_error_detail_validation_list() -> None:
    request = httpx.Request("POST", "http://localhost:8000/api/v1/estimate")
    response = httpx.Response(
        422,
        request=request,
        json={
            "detail": [
                {
                    "loc": ["body", "description"],
                    "msg": "String should have at least 10 characters",
                }
            ]
        },
    )
    text = parse_error_detail(response)
    assert "description" in text
    assert "10 characters" in text


def test_get_api_root_url_strips_api_v1_suffix() -> None:
    assert get_api_root_url("http://localhost:8000/api/v1") == "http://localhost:8000"


def test_strategy_catalog_has_eight_entries_in_canonical_order() -> None:
    names = [entry["name"] for entry in STRATEGY_CATALOG]
    assert names == [
        "structural",
        "fixed_size",
        "recursive",
        "sentence_window",
        "hierarchical",
        "semantic",
        "propositional",
        "contextual_retrieval",
    ]


def test_label_for_and_cost_hint() -> None:
    assert label_for("structural") == "Structural (JSON)"
    assert "seconds" in cost_hint(["structural", "recursive"]).lower()
    assert "$0.15" in cost_hint(["propositional"])


def test_build_settings_update_payload_maps_empty_to_none() -> None:
    payload = build_settings_update_payload(
        {"PRIMARY_MODEL": "gpt-4o", "FALLBACK_MODEL": ""}
    )
    assert payload == {"PRIMARY_MODEL": "gpt-4o", "FALLBACK_MODEL": None}


def test_build_retrieval_update_payload_includes_s11_flags() -> None:
    payload = build_retrieval_update_payload(
        search_mode=None,
        rerank=None,
        routing_enabled=None,
        query_transform_enabled=None,
        temporal_decay_enabled=None,
        task_hours_top_k=None,
        task_hours_distance_threshold=None,
        hallucination_gate_enabled=False,
        augmentation_enabled=True,
        synthesis_enabled=True,
        touched={
            "hallucination_gate_enabled",
            "augmentation_enabled",
            "synthesis_enabled",
        },
    )
    assert payload == {
        "hallucination_gate_enabled": False,
        "augmentation_enabled": True,
        "synthesis_enabled": True,
    }


def test_fetch_available_agent_models_uses_api_catalog_and_filters_non_openai(
    monkeypatch,
) -> None:
    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {
                "available_models": [
                    "gpt-5",
                    "anthropic/claude-sonnet-4-5",
                    "claude-3-5-sonnet",
                    "openai/gpt-4o-mini",
                    "gemini/gemini-2.5-pro",
                ]
            }

    calls = []

    def fake_get(url, *, timeout):
        calls.append((url, timeout))
        return Response()

    fetch_available_agent_models.clear()
    monkeypatch.setattr("streamlit_ui.common.httpx.get", fake_get)
    models = fetch_available_agent_models("http://api/api/v1", timeout=1.25)

    assert models == ["gpt-5", "openai/gpt-4o-mini"]
    assert calls == [("http://api/api/v1/config/models", 1.25)]


def test_fetch_available_agent_models_falls_back_on_timeout_or_api_error(
    monkeypatch,
) -> None:
    def fail(*args, **kwargs):
        raise httpx.ReadTimeout("catalog unavailable")

    fetch_available_agent_models.clear()
    monkeypatch.setattr("streamlit_ui.common.httpx.get", fail)
    assert fetch_available_agent_models("http://api") == list(DEFAULT_AGENT_MODELS)


def test_fetch_available_agent_models_keeps_saved_model_absent_from_catalog(
    monkeypatch,
) -> None:
    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {"available_models": ["gpt-5"]}

    fetch_available_agent_models.clear()
    monkeypatch.setattr("streamlit_ui.common.httpx.get", lambda *a, **k: Response())
    models = fetch_available_agent_models("http://api", saved_model="gpt-retired")

    assert models == ["gpt-retired", "gpt-5"]
    assert agent_model_label("gpt-retired", ["gpt-5"]) == (
        "gpt-retired (no disponible)"
    )
