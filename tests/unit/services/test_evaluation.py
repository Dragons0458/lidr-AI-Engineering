from app.schemas.estimation import OutputFormat
from app.services.evaluation import evaluate_estimation_structure


def test_evaluates_well_formed_v1_phases_table() -> None:
    text = """## Estimacion: Portal interno

| Phase | Tasks | Hours | Team |
|---|---|---:|---|
| Discovery | Scope | 10 | PM |
| Backend | API | 40 | Backend Engineer |
| Frontend | UI | 30 | Frontend Engineer |
| Total | - | 80 | Core team |

Total estimated hours: 80h
Equipo recomendado: PM, Backend Engineer, Frontend Engineer
Duracion estimada: 4 semanas
"""

    result = evaluate_estimation_structure(text, "stop", OutputFormat.PHASES_TABLE)

    assert result.score == 1.0
    assert result.has_breakdown_table is True
    assert result.declared_total_hours == 80
    assert result.sum_row_hours == 80
    assert result.hours_match is True
    assert result.issues == []


def test_evaluates_well_formed_v2_phases_table() -> None:
    text = """## Estimacion: Checkout

| Phase | Scope | Base Hours | Buffer Hours | Team |
|---|---|---:|---:|---|
| Discovery | KPIs | 10 | 2 | PM |
| Backend | Payments | 44 | 8 | Backend Engineer |
| Frontend | Checkout | 40 | 6 | Frontend Engineer |
| Total | - | 94 | 16 | Core team |

Total planned hours: 110h
Equipo recomendado: PM, Backend Engineer, Frontend Engineer
Suggested timeline: 5 weeks
"""

    result = evaluate_estimation_structure(text, "end_turn", OutputFormat.PHASES_TABLE)

    assert result.score == 1.0
    assert result.declared_total_hours == 110
    assert result.sum_row_hours == 110
    assert result.hours_match is True


def test_flags_hours_mismatch() -> None:
    text = """## Estimacion: Portal

| Phase | Tasks | Hours | Team |
|---|---|---:|---|
| Backend | API | 40 | Backend Engineer |
| Frontend | UI | 30 | Frontend Engineer |
| Total | - | 90 | Core team |

Total estimated hours: 90h
Equipo recomendado: Backend Engineer, Frontend Engineer
Duracion estimada: 4 semanas
"""

    result = evaluate_estimation_structure(text, "stop", OutputFormat.PHASES_TABLE)

    assert result.hours_match is False
    assert "Total hours mismatch" in result.issues[-1]


def test_flags_missing_table_and_length_finish_reason() -> None:
    result = evaluate_estimation_structure(
        "## Estimacion\nTotal estimated hours: 20h",
        "length",
        OutputFormat.PHASES_TABLE,
    )

    assert result.has_breakdown_table is False
    assert result.finish_reason_ok is False
    assert "Missing phases table" in result.issues[0]
    assert "finish_reason='length'" in result.issues[-1]


def test_empty_text_scores_zero() -> None:
    result = evaluate_estimation_structure("", "length", OutputFormat.LINE_ITEMS)

    assert result.score == 0.0
    assert result.issues


def test_narrative_uses_laxer_checks() -> None:
    text = (
        "Scope summary: Plataforma interna de reportes.\n"
        "Effort breakdown: backend (40h), frontend (30h).\n"
        "Total estimated hours: 70h\n"
        "Equipo recomendado: Backend Engineer y Frontend Engineer.\n"
        "Suggested timeline: 4 weeks."
    )

    result = evaluate_estimation_structure(text, "tool_use", OutputFormat.NARRATIVE)

    assert result.score == 1.0
    assert result.has_breakdown_table is False
    assert result.hours_match is None
