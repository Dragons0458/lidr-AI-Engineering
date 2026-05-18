from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.schemas.estimation import EstimationRequest

MessageRole = Literal["system", "user", "assistant", "tool"]
HistoryMessageRole = Literal["user", "assistant", "tool"]
PromptVersion = Literal["v1", "v2"]
MAX_TURNS = 6


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class ConversationTurn(BaseModel):
    user: ChatMessage | None = None
    assistant: ChatMessage | None = None
    tool_messages: list[ChatMessage] = Field(default_factory=list)

    @property
    def messages(self) -> list[ChatMessage]:
        turn_messages = []

        if self.user:
            turn_messages.append(self.user)
        if self.assistant:
            turn_messages.append(self.assistant)

        turn_messages.extend(self.tool_messages)
        return turn_messages


@dataclass
class ConversationHistory:
    """Volatile chat history with a sliding window.

    The service is still in a single-process phase, so accepting loss of this
    state on restart keeps the implementation simple until persistence is a real
    product requirement.
    """

    max_turns: int = MAX_TURNS
    turns: deque[ConversationTurn] = field(init=False)

    def __post_init__(self) -> None:
        self.turns = deque(maxlen=max(self.max_turns, 0))

    def add_message(self, role: HistoryMessageRole, content: str) -> None:
        message = ChatMessage(role=role, content=content)

        if role == "user":
            self.turns.append(ConversationTurn(user=message))
            return

        if not self.turns:
            self.turns.append(ConversationTurn())
            if not self.turns:
                return

        if role == "assistant":
            self.turns[-1].assistant = message
            return

        self.turns[-1].tool_messages.append(message)

    def to_messages_list(
        self,
        request: "EstimationRequest",
        *,
        prompt_version: PromptVersion = "v1",
        project_metadata: "ProjectMetadata | None" = None,
    ) -> list[dict[str, str]]:
        """Build API-ready messages with a fresh system prompt.

        The system prompt is intentionally rendered at call time so mutable
        project metadata cannot leave the conversation with stale instructions.
        """
        from app.prompts.loader import render_estimation_prompt

        system_prompt, user_prompt = render_estimation_prompt(
            request, version=prompt_version, project_metadata=project_metadata
        )
        messages = [ChatMessage(role="system", content=system_prompt)]

        for turn in self.turns:
            messages.extend(turn.messages)

        messages.append(ChatMessage(role="user", content=user_prompt))
        return [message.model_dump() for message in messages]


class ProjectMetadata(BaseModel):
    """Mutable, in-memory project facts inferred during a session.

    This metadata is intentionally process-local for now: early iterations value
    fast feedback over durability, and losing it on deploy/restart is acceptable
    until multi-worker or long-lived sessions are required.
    """

    project_name: str | None = None
    assumed_team_size: int | None = Field(default=None, ge=1)
    mentioned_technologies: list[str] = Field(default_factory=list)
    excluded_technologies: list[str] = Field(default_factory=list)
    agreed_scope: str | None = None


@dataclass
class Session:
    """Process-local session container indexed by ``session_id``.

    A plain dictionary is enough at this stage because the API is not relying on
    cross-process continuity. Persistence can be added later behind this class
    without changing callers that use ``get_or_create``.
    """

    session_id: str
    history: ConversationHistory = field(
        default_factory=lambda: ConversationHistory(max_turns=_default_max_turns())
    )
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)

    _sessions: ClassVar[dict[str, "Session"]] = {}

    @classmethod
    def get_or_create(cls, session_id: str, max_turns: int | None = None) -> "Session":
        if session_id not in cls._sessions:
            cls._sessions[session_id] = cls(
                session_id=session_id,
                history=ConversationHistory(
                    max_turns=max_turns
                    if max_turns is not None
                    else _default_max_turns()
                ),
            )
        return cls._sessions[session_id]

    @classmethod
    def get(cls, session_id: str) -> "Session | None":
        return cls._sessions.get(session_id)

    @classmethod
    def delete(cls, session_id: str) -> None:
        cls._sessions.pop(session_id, None)

    @classmethod
    def clear_all(cls) -> None:
        cls._sessions.clear()


def _default_max_turns() -> int:
    from app.config import get_settings

    return get_settings().CONVERSATION_MAX_TURNS
