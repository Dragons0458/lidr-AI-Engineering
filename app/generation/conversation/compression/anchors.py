"""Heuristic and LLM-based anchor detection for durable conversation facts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import instructor
import structlog
from pydantic import BaseModel, Field

from app.generation.conversation.store import ChatMessage

if TYPE_CHECKING:
    from app.foundation.llm.wrapper import LLMWrapper

log = structlog.get_logger()

_HEURISTIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("nda", re.compile(r"\b(nda|non[- ]disclosure|confidencial)\b", re.I)),
    (
        "signed_contract",
        re.compile(r"\b(contrato\s+firmado|signed\s+contract)\b", re.I),
    ),
    (
        "frozen_scope",
        re.compile(
            r"\b(scope\s+frozen|alcance\s+congelado|alcance\s+cerrado)\b",
            re.I,
        ),
    ),
    (
        "locked_budget",
        re.compile(
            r"\b(budget\s+locked|presupuesto\s+cerrado|presupuesto\s+bloqueado)\b",
            re.I,
        ),
    ),
    (
        "compliance",
        re.compile(r"\b(hipaa|gdpr|sox|pci[- ]dss|iso[- ]?27001|ccpa)\b", re.I),
    ),
    (
        "hard_deadline",
        re.compile(
            r"\b(hard\s+deadline|fecha\s+l[ií]mite|deadline\s+inamovible)\b",
            re.I,
        ),
    ),
    (
        "explicit_commitment",
        re.compile(
            r"\b(we\s+agreed|acordamos|firmamos|compromiso\s+expl[ií]cito)\b",
            re.I,
        ),
    ),
)


class _AnchorClassification(BaseModel):
    is_anchor: bool
    matched_rules: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class AnchorMatch:
    is_anchor: bool
    matched_rules: list[str]


class AnchorDetector:
    def __init__(
        self,
        mode: Literal["heuristic", "llm"] = "heuristic",
        *,
        llm_wrapper: LLMWrapper | None = None,
        llm_model: str | None = None,
    ) -> None:
        self.mode = mode
        self.llm_wrapper = llm_wrapper
        self.llm_model = llm_model
        self._instructor_client = None
        if mode == "llm" and llm_wrapper is not None:
            from app.foundation.llm.wrapper import completion

            self._instructor_client = instructor.from_litellm(completion)

    def detect(self, message: ChatMessage) -> AnchorMatch:
        if self.mode == "llm":
            return self._detect_llm(message)
        return self._detect_heuristic(message)

    def _detect_heuristic(self, message: ChatMessage) -> AnchorMatch:
        matched = [
            name
            for name, pattern in _HEURISTIC_PATTERNS
            if pattern.search(message.content)
        ]
        return AnchorMatch(is_anchor=bool(matched), matched_rules=matched)

    def _detect_llm(self, message: ChatMessage) -> AnchorMatch:
        if self._instructor_client is None or not self.llm_model:
            return self._detect_heuristic(message)
        try:
            result = self._instructor_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Classify whether this user message is a durable anchor "
                            "(NDA, signed contract, frozen scope, locked budget, "
                            "compliance, hard deadline, explicit commitment). "
                            "Reply with structured JSON only."
                        ),
                    },
                    {"role": "user", "content": message.content},
                ],
                response_model=_AnchorClassification,
                max_tokens=200,
                max_retries=1,
                temperature=0,
            )
            return AnchorMatch(
                is_anchor=result.is_anchor,
                matched_rules=result.matched_rules,
            )
        except Exception as exc:
            log.warning("anchor_llm_detection_failed", error=str(exc))
            return self._detect_heuristic(message)
