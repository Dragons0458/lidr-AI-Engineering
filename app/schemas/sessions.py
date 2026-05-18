from pydantic import BaseModel, ConfigDict


class SessionCreateRequest(BaseModel):
    """Request body for creating an in-memory session.

    The model is intentionally empty for now because the server generates the
    session identifier and keeps the session state process-local.
    """

    model_config = ConfigDict(extra="forbid")


class SessionResponse(BaseModel):
    session_id: str
