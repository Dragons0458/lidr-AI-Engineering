from pydantic import BaseModel, Field


class AttachmentText(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
