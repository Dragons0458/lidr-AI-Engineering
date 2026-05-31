"""Actor-Critic-Boss orchestration without additional LLM calls."""

from __future__ import annotations

from collections.abc import Callable

from app.schemas.acb import ACBIteration, BossDecision, BossTrace
from app.schemas.critic import CriticFeedback, CriticIssue
from app.schemas.estimation import (
    LOW_CONFIDENCE_THRESHOLD,
    EstimationResult,
)

ActorCallable = Callable[[CriticFeedback | None], EstimationResult]
CriticCallable = Callable[[EstimationResult], CriticFeedback]


class Boss:
    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations

    def run(
        self,
        *,
        actor: ActorCallable,
        critic: CriticCallable,
    ) -> tuple[EstimationResult, BossTrace]:
        iterations: list[ACBIteration] = []
        last_result: EstimationResult | None = None
        last_feedback: CriticFeedback | None = None
        critic_feedback: CriticFeedback | None = None

        for iteration in range(1, self.max_iterations + 1):
            draft = actor(critic_feedback)
            last_result = draft
            review = critic(draft)
            last_feedback = review
            iterations_left = self.max_iterations - iteration
            decision = self._decide(review, iterations_left)

            iterations.append(
                ACBIteration(
                    iteration=iteration,
                    decision_after=decision,
                    critic_verdict=review.verdict,
                    critic_confidence=review.confidence_in_review,
                    issue_summary=[
                        f"{issue.severity}/{issue.category}: {issue.description}"
                        for issue in review.issues[:6]
                    ],
                )
            )

            if decision == "accept":
                return draft, BossTrace(
                    iterations=iterations,
                    final_decision="accept",
                    iterations_run=iteration,
                )

            if decision == "synthesize":
                synthesized = self._synthesize_fallback(draft, review)
                return synthesized, BossTrace(
                    iterations=iterations,
                    final_decision="synthesize",
                    iterations_run=iteration,
                )

            critic_feedback = review

        assert last_result is not None
        fallback_feedback = last_feedback or CriticFeedback(
            verdict="accept",
            issues=[],
            confidence_in_review=0,
        )
        synthesized = self._synthesize_fallback(last_result, fallback_feedback)
        return synthesized, BossTrace(
            iterations=iterations,
            final_decision="synthesize",
            iterations_run=self.max_iterations,
        )

    @staticmethod
    def _decide(review: CriticFeedback, iterations_left: int) -> BossDecision:
        if review.verdict == "accept":
            return "accept"
        if review.verdict == "reject" or iterations_left <= 0:
            return "synthesize"
        if review.verdict == "needs_iteration" and iterations_left > 0:
            return "iterate"
        return "synthesize"

    @staticmethod
    def _synthesize_fallback(
        last_result: EstimationResult,
        last_feedback: CriticFeedback,
    ) -> EstimationResult:
        open_issues = [
            issue
            for issue in last_feedback.issues
            if issue.severity in ("critical", "major")
        ]
        caveat_lines = _format_caveats(open_issues)
        summary = last_result.summary
        if caveat_lines:
            summary = f"{summary}\n\nCaveats:\n" + "\n".join(caveat_lines)

        reduced_confidence = max(
            LOW_CONFIDENCE_THRESHOLD,
            min(last_result.confidence_pct, 50) - 5 * len(open_issues),
        )
        return last_result.model_copy(
            update={
                "summary": summary[:1200],
                "confidence_pct": reduced_confidence,
            }
        )


def _format_caveats(issues: list[CriticIssue]) -> list[str]:
    lines: list[str] = []
    for issue in issues[:8]:
        line = f"- [{issue.category}] {issue.description}"
        if issue.suggested_fix:
            line += f" (suggested: {issue.suggested_fix})"
        lines.append(line)
    return lines
