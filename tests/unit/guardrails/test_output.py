from app.guardrails.output import enforce_output


def test_normal_text_unchanged() -> None:
    text = "## Estimacion\nTotal estimated hours: 40h"
    result = enforce_output(text)

    assert result.text == text
    assert result.out_of_scope is False
    assert result.pii_redacted is False


def test_out_of_scope_prefix() -> None:
    text = "Out of scope: not a software project\n\n| Phase | Hours |\n"
    result = enforce_output(text)

    assert result.out_of_scope is True
    assert result.text == "Out of scope: not a software project"
    assert "| Phase" not in result.text


def test_pii_redacted_without_raising() -> None:
    text = "Contacto user@example.com y telefono +34 612 345 678."
    result = enforce_output(text)

    assert "[REDACTED]" in result.text
    assert result.pii_redacted is True
    assert "user@example.com" not in result.text


def test_enforce_output_never_raises() -> None:
    enforce_output("")
    enforce_output("Out of scope: vague request")
