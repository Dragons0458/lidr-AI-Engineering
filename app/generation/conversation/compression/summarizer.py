"""Cumulative conversation summarization for evicted turns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from app.foundation.prompts.loader import render_conversation_summary_prompt
from app.generation.conversation.store import ChatMessage

if TYPE_CHECKING:
    from app.foundation.llm.wrapper import LLMWrapper

log = structlog.get_logger()


class _SummaryEnvelope(BaseModel):
    summary: str = Field(min_length=1, max_length=4000)


class CumulativeSummarizer:
    def __init__(self, llm_wrapper: LLMWrapper, model: str) -> None:
        self.llm_wrapper = llm_wrapper
        self.model = model

    def summarize(
        self,
        *,
        previous_summary: str | None,
        evicted: list[ChatMessage],
    ) -> str:
        if not evicted:
            return previous_summary or ""

        system_prompt, user_prompt = render_conversation_summary_prompt(
            previous_summary=previous_summary or "",
            evicted=evicted,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            envelope, _meta = self.llm_wrapper.complete_structured_chat(
                messages=messages,
                response_model=_SummaryEnvelope,
                model=self.model,
                max_tokens=1200,
                max_retries=2,
                temperature=0,
                use_cache=False,
            )
            return envelope.summary
        except Exception as exc:
            log.warning("summarizer_failed", error=str(exc))
            return previous_summary or ""
