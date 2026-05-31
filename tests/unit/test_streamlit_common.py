import httpx

from streamlit_common import (
    format_api_error,
    format_guardrail_detail,
    parse_error_detail,
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
