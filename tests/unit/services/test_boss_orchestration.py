from app.schemas.critic import CriticFeedback, CriticIssue
from app.schemas.estimation import EstimationResult, Phase
from app.services.boss import Boss


def _draft(confidence: int = 80) -> EstimationResult:
    return EstimationResult(
        summary="Portal SaaS con autenticación y panel de reportes para el cliente.",
        confidence_pct=confidence,
        phases=[
            Phase(
                name="Backend",
                base_hours=40,
                buffer_hours=5,
                team="2 devs",
                summary="API y autenticación con pruebas unitarias básicas incluidas.",
            )
        ],
        total_base_hours=40,
        total_buffer_hours=5,
        total_hours=45,
        total_cost_eur=3000,
    )


def _accept() -> CriticFeedback:
    return CriticFeedback(verdict="accept", issues=[], confidence_in_review=90)


def _needs_iteration() -> CriticFeedback:
    return CriticFeedback(
        verdict="needs_iteration",
        issues=[
            CriticIssue(
                category="math_error",
                severity="major",
                field_path="total_hours",
                description="Totals do not match phases.",
                suggested_fix="Recompute totals.",
            )
        ],
        confidence_in_review=70,
    )


def test_accept_on_first_iteration() -> None:
    calls = {"actor": 0, "critic": 0}

    def actor(_feedback):
        calls["actor"] += 1
        return _draft()

    def critic(_draft):
        calls["critic"] += 1
        return _accept()

    result, trace = Boss(max_iterations=3).run(actor=actor, critic=critic)
    assert trace.final_decision == "accept"
    assert trace.iterations_run == 1
    assert calls == {"actor": 1, "critic": 1}
    assert result.confidence_pct == 80


def test_iterate_then_accept() -> None:
    critic_calls = 0

    def actor(_feedback):
        return _draft()

    def critic(_draft):
        nonlocal critic_calls
        critic_calls += 1
        return _needs_iteration() if critic_calls == 1 else _accept()

    result, trace = Boss(max_iterations=3).run(actor=actor, critic=critic)
    assert trace.final_decision == "accept"
    assert trace.iterations_run == 2
    assert len(trace.iterations) == 2
    assert result.total_hours == 45


def test_budget_exhausted_synthesizes_with_reduced_confidence() -> None:
    def actor(_feedback):
        return _draft(confidence=70)

    def critic(_draft):
        return _needs_iteration()

    result, trace = Boss(max_iterations=1).run(actor=actor, critic=critic)
    assert trace.final_decision == "synthesize"
    assert "Caveats:" in result.summary
    assert result.confidence_pct <= 70


def test_reject_synthesizes_immediately() -> None:
    def actor(_feedback):
        return _draft()

    def critic(_draft):
        return CriticFeedback(
            verdict="reject",
            issues=[
                CriticIssue(
                    category="scope_mismatch",
                    severity="critical",
                    field_path="summary",
                    description="Not a software project.",
                )
            ],
            confidence_in_review=95,
        )

    _, trace = Boss(max_iterations=3).run(actor=actor, critic=critic)
    assert trace.final_decision == "synthesize"
    assert trace.iterations_run == 1
