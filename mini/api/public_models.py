"""The request and response bodies of the API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AttachmentOut(BaseModel):
    """A file sent with a message. Download it from GET /attachments/{id}."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    content_type: str
    size: int  # bytes


class VoiceNoteOut(BaseModel):
    """Set on a message that's a voice note; its content is the transcript."""

    model_config = ConfigDict(from_attributes=True)

    attachment_id: int  # the recording (also in the message's attachments)
    duration: float | None  # seconds


class PersonOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str  # their canonical name


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: Literal["user", "assistant"]
    content: str
    parent_id: int | None  # the top-level message whose thread this is in
    created_at: datetime
    attachments: list[AttachmentOut] = []
    voice_note: VoiceNoteOut | None = None
    sender: PersonOut | None = None  # who sent a user message, if known


class SenderIn(BaseModel):
    """The platform account sending, which must be linked to a person."""

    provider: str = Field(min_length=1)  # e.g. "discord", "web"
    external_id: str = Field(min_length=1)  # their id on that platform


class VoiceNoteIn(BaseModel):
    attachment_id: int  # the uploaded recording; the server transcribes it
    duration: float | None = None  # seconds


class SendMessageIn(BaseModel):
    content: str = ""
    parent_id: int | None = None  # reply in this message's thread
    # Files uploaded with POST /attachments to send with the message.
    attachment_ids: list[int] = []
    # Send a voice note instead of text: its transcript becomes the content.
    voice_note: VoiceNoteIn | None = None
    sender: SenderIn | None = None

    @model_validator(mode="after")
    def _check(self) -> "SendMessageIn":
        if self.voice_note is not None and self.content.strip():
            raise ValueError("A voice note's content is its transcript, so leave content empty")
        if not self.content.strip() and not self.attachment_ids and self.voice_note is None:
            raise ValueError("A message needs content, attachments or a voice note")
        return self


class MessagePage(BaseModel):
    page: int
    page_size: int
    total: int
    messages: list[ChatMessageOut]  # newest first
