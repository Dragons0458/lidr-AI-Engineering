"""LLM critic for structured estimation drafts."""

from __future__ import annotations

import structlog

from app.foundation.prompts.loader import render_critic_prompt
from app.domain.schemas.critic import CriticFeedback
from app.domain.schemas.estimation import EstimationResult
from app.foundation.llm.wrapper import LLMWrapper
from app.generation.conversation.store import ProjectMetadata
from app.generation.conversation.tier_resolver import Tier

log = structlog.get_logger()


class Critic:
    def __init__(self, llm_wrapper: LLMWrapper, model: str) -> None:
        self.llm_wrapper = llm_wrapper
        self.model = model

    def review(
        self,
        *,
        transcript: str,
        metadata: ProjectMetadata,
        tier: Tier,
        result: EstimationResult,
    ) -> CriticFeedback:
        system_prompt, user_prompt = render_critic_prompt(
            transcript=transcript,
            metadata=metadata,
            tier=tier.value,
            result=result,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            feedback, _meta = self.llm_wrapper.complete_structured_chat(
                messages=messages,
                response_model=CriticFeedback,
                model=self.model,
                max_tokens=2000,
                max_retries=2,
                temperature=0,
            )
            return feedback
        except Exception as exc:
            log.warning("critic_failed_fallback_accept", error=str(exc))
            return CriticFeedback(
                verdict="accept",
                issues=[],
                confidence_in_review=0,
            )
