"""The request and response bodies of the API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class AttachmentOut(BaseModel):
    """A file sent with a message. Download it from GET /attachments/{id}."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    content_type: str
    size: int  # bytes


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: Literal["user", "assistant"]
    content: str
    parent_id: int | None  # the top-level message whose thread this is in
    created_at: datetime
    attachments: list[AttachmentOut] = []


class SendMessageIn(BaseModel):
    content: str = ""
    parent_id: int | None = None  # reply in this message's thread
    # Files uploaded with POST /attachments to send with the message.
    attachment_ids: list[int] = []

    @model_validator(mode="after")
    def _not_empty(self) -> "SendMessageIn":
        if not self.content.strip() and not self.attachment_ids:
            raise ValueError("A message needs content or attachments")
        return self


class MessagePage(BaseModel):
    page: int
    page_size: int
    total: int
    messages: list[ChatMessageOut]  # newest first
