"""The request and response bodies of the API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: Literal["user", "assistant"]
    content: str
    parent_id: int | None  # the top-level message whose thread this is in
    created_at: datetime


class SendMessageIn(BaseModel):
    content: str = Field(min_length=1)
    parent_id: int | None = None  # reply in this message's thread


class MessagePage(BaseModel):
    page: int
    page_size: int
    total: int
    messages: list[ChatMessageOut]  # newest first
