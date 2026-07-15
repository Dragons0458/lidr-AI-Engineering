"""Shared schemas for auditable agent traces."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class AgentStep(BaseModel):
    """One reason-act-observe step of an agent run."""

    step: int = Field(ge=1)
    reasoning_summary: str | None = None
    tool: str
    tool_args: dict[str, Any]
    observation: str


class AgentTrace(BaseModel):
    """Ordered record of the actions performed by an agent."""

    steps: list[AgentStep] = Field(default_factory=list)

    def render(self) -> str:
        """Render a stable, control-character-safe console representation."""
        if not self.steps:
            return "(no tool steps — the agent answered without calling any tool)"
        blocks: list[str] = []
        for step in self.steps:
            reasoning = _CONTROL_CHARS.sub(
                "", step.reasoning_summary or "(no reasoning summary emitted)"
            )
            args = _CONTROL_CHARS.sub(
                "", json.dumps(step.tool_args, ensure_ascii=False, default=str)
            )
            observation = _CONTROL_CHARS.sub("", step.observation)
            tool = _CONTROL_CHARS.sub("", step.tool)
            blocks.append(
                f"STEP {step.step}\n"
                f"  reasoning:   {reasoning}\n"
                f"  action:      {tool}({args})\n"
                f"  observation: {observation}"
            )
        return "\n\n".join(blocks)
