from pydantic import BaseModel, ConfigDict, Field


class SessionCreateRequest(BaseModel):
    """Request body for creating an in-memory session.

    The model is intentionally empty for now because the server generates the
    session identifier and keeps the session state process-local.
    """

    model_config = ConfigDict(extra="forbid")


class SessionResponse(BaseModel):
    session_id: str


class SessionDebugResponse(BaseModel):
    session_id: str
    message_count: int
    anchors_count: int
    summary_chars: int
    last_resolved_tier: str | None = None
    last_tier_rule: str | None = None
    summary: str = ""
    anchors: list[str] = Field(default_factory=list)
    metadata: dict[str, object]
    last_turn_observation: dict[str, object] | None = None
