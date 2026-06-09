"""Build enriched transcripts for multi-turn session estimation."""

from __future__ import annotations

import json

from app.domain.schemas.estimation import EstimationRequest, EstimationResult
from app.generation.conversation.store import ChatMessage, ConversationHistory, Session

_SUMMARY_LABEL = "[Resumen de la conversación previa]"


def format_assistant_message_for_context(content: str) -> str:
    """Render assistant output as readable context (JSON estimations → short summary)."""
    stripped = content.strip()
    if not stripped:
        return ""
    if stripped.startswith("{"):
        try:
            result = EstimationResult.model_validate_json(stripped)
        except (json.JSONDecodeError, ValueError):
            return stripped
        if result.summary.startswith("Out of scope:"):
            return result.summary
        return (
            f"{result.summary} "
            f"(totales: {result.total_hours}h, "
            f"{result.total_cost_eur} EUR, confianza {result.confidence_pct}%)"
        )
    return stripped


def format_messages_for_context(messages: list[ChatMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        if message.role == "system":
            continue
        label = {"user": "Usuario", "assistant": "Asistente", "tool": "Tool"}.get(
            message.role, message.role
        )
        body = message.content
        if message.role == "assistant":
            body = format_assistant_message_for_context(body)
        if body:
            lines.append(f"{label}: {body}")
    return "\n\n".join(lines)


def build_conversation_context(
    history: ConversationHistory,
    *,
    exclude_latest_user: bool = True,
) -> str:
    """Format summary, anchors, and prior turns for prompt injection."""
    blocks: list[str] = []

    if history.summary:
        blocks.append(f"{_SUMMARY_LABEL}\n{history.summary}")

    if history.anchors:
        anchor_lines = [
            format_messages_for_context([anchor]) for anchor in history.anchors
        ]
        blocks.append(
            "Hechos ancla preservados:\n"
            + "\n".join(line for line in anchor_lines if line)
        )

    turns = list(history.turns)
    if exclude_latest_user and turns and turns[-1].user is not None:
        turns = turns[:-1]

    for index, turn in enumerate(turns, start=1):
        turn_text = format_messages_for_context(turn.messages)
        if turn_text:
            blocks.append(f"--- Turno {index} ---\n{turn_text}")

    return "\n\n".join(block for block in blocks if block.strip())


def build_acb_turn_context(
    session: Session,
    request: EstimationRequest,
) -> tuple[str, str, bool]:
    """Return (enriched_transcript, conversation_context, is_follow_up)."""
    attachments_text = "\n".join(
        attachment.content for attachment in request.attachments or []
    )
    conversation_context = build_conversation_context(
        session.history,
        exclude_latest_user=False,
    )
    is_follow_up = bool(conversation_context.strip())

    latest_parts = [request.description.strip()]
    if attachments_text:
        latest_parts.append(attachments_text)
    latest_block = "\n\n".join(part for part in latest_parts if part)

    if is_follow_up:
        enriched = "\n\n".join(
            part
            for part in (
                conversation_context,
                f"Mensaje actual del usuario:\n{latest_block}",
            )
            if part.strip()
        )
    else:
        enriched = latest_block

    return enriched, conversation_context, is_follow_up
