import httpx

from streamlit_ui.common import (
    format_api_error,
    format_guardrail_detail,
    get_api_root_url,
    parse_error_detail,
)
from streamlit_ui.rag import (
    STRATEGY_CATALOG,
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
