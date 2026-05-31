from unittest.mock import MagicMock

from app.schemas.estimation import EstimationResult, Phase
from app.services.critic import Critic
from app.services.sessions import ProjectMetadata
from app.services.tier_resolver import Tier


def _result() -> EstimationResult:
    return EstimationResult(
        summary="Estimación de portal con login y reportes para usuarios internos.",
        confidence_pct=80,
        phases=[
            Phase(
                name="Backend",
                base_hours=30,
                buffer_hours=5,
                team="2 devs",
                summary="API REST con autenticación y pruebas básicas del módulo.",
            )
        ],
        total_base_hours=30,
        total_buffer_hours=5,
        total_hours=35,
        total_cost_eur=2000,
    )


def test_llm_failure_returns_accept_degraded() -> None:
    wrapper = MagicMock()
    wrapper.complete_structured_chat.side_effect = RuntimeError("down")
    feedback = Critic(wrapper, "gpt-4o-mini").review(
        transcript="Build portal",
        metadata=ProjectMetadata(),
        tier=Tier.DEFAULT,
        result=_result(),
    )
    assert feedback.verdict == "accept"
    assert feedback.issues == []
    assert feedback.confidence_in_review == 0
