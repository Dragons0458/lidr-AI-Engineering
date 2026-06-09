"""Compression policy: promote anchors, summarize evicted non-anchor turns."""

from __future__ import annotations

from typing import Literal

import structlog

from app.generation.conversation.compression.anchors import AnchorDetector
from app.generation.conversation.compression.summarizer import CumulativeSummarizer
from app.generation.conversation.store import ChatMessage, ConversationHistory
from app.foundation.llm.wrapper import LLMWrapper

log = structlog.get_logger()


class CompressionPolicy:
    def __init__(
        self,
        anchor_detector: AnchorDetector,
        summarizer: CumulativeSummarizer,
    ) -> None:
        self.anchor_detector = anchor_detector
        self.summarizer = summarizer

    def should_compress(self, history: ConversationHistory) -> bool:
        return len(history.turns) > history.max_turns

    def apply(self, history: ConversationHistory) -> None:
        if not self.should_compress(history):
            return

        evicted_for_summary: list[ChatMessage] = []
        while len(history.turns) > history.max_turns:
            oldest = history.turns.popleft()
            user_msg = oldest.user
            if user_msg is None:
                continue

            match = self.anchor_detector.detect(user_msg)
            if match.is_anchor:
                history.anchors.append(user_msg)
                if oldest.assistant is not None:
                    history.anchors.append(oldest.assistant)
                history.anchors.extend(oldest.tool_messages)
            else:
                evicted_for_summary.extend(oldest.messages)

        if evicted_for_summary:
            history.summary = self.summarizer.summarize(
                previous_summary=history.summary,
                evicted=evicted_for_summary,
            )

        log.info(
            "history_compressed",
            anchors_count=history.anchors_count,
            summary_chars=history.summary_chars,
            turns_remaining=len(history.turns),
        )


def apply_compression(
    history: ConversationHistory,
    *,
    llm_wrapper: LLMWrapper,
    compression_model: str,
    anchor_detection_mode: Literal["heuristic", "llm"] = "heuristic",
) -> None:
    detector = AnchorDetector(
        anchor_detection_mode,
        llm_wrapper=llm_wrapper,
        llm_model=compression_model,
    )
    summarizer = CumulativeSummarizer(llm_wrapper, compression_model)
    CompressionPolicy(detector, summarizer).apply(history)
