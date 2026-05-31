import pytest
from pydantic import ValidationError

from app.schemas.estimation import (
    OUT_OF_SCOPE_PREFIX,
    EstimationResult,
    Phase,
)


def _valid_result(**updates) -> EstimationResult:
    base = dict(
        summary="Entrega de portal SaaS con autenticación y reportes.",
        confidence_pct=75,
        phases=[
            Phase(
                name="Backend",
                base_hours=80,
                buffer_hours=10,
                team="2 backend",
                summary="API REST y autenticación con pruebas básicas.",
            )
        ],
        total_base_hours=80,
        total_buffer_hours=10,
        total_hours=90,
        total_cost_eur=5000,
    )
    base.update(updates)
    return EstimationResult(**base)


def test_totals_must_match_phases() -> None:
    with pytest.raises(ValidationError):
        _valid_result(total_base_hours=99)


def test_low_confidence_requires_out_of_scope_prefix() -> None:
    with pytest.raises(ValidationError):
        _valid_result(confidence_pct=10, summary="Too vague to estimate.")


def test_low_confidence_with_prefix_is_valid() -> None:
    result = _valid_result(
        confidence_pct=10,
        summary=f"{OUT_OF_SCOPE_PREFIX} Not a software project.",
        phases=[
            Phase(
                name="Triage",
                base_hours=0,
                buffer_hours=1,
                team="PM",
                summary="Clasificación inicial: proyecto fuera de alcance de software.",
            )
        ],
        total_hours=1,
        total_base_hours=0,
        total_buffer_hours=1,
    )
    assert result.summary.startswith(OUT_OF_SCOPE_PREFIX)
