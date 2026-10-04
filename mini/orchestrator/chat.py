"""The user's chat with the orchestrator.

Anything that shows the chat (the API's event stream, for one) can `subscribe`
to get every new message as it's saved.
"""

import asyncio
from collections.abc import Collection
from typing import Literal

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from mini import attachments, people
from mini.database.models import ChatMessage, VoiceNote

_subscribers: set[asyncio.Queue[ChatMessage]] = set()


def send_message(
    engine: Engine,
    role: Literal["user", "assistant"],
    content: str,
    parent_id: int | None = None,
    attachment_ids: Collection[int] = (),
    voice_note: tuple[int, float | None] | None = None,
    sender: tuple[str, str] | None = None,
) -> ChatMessage:
    """Add a message to the chat. `parent_id` must be a top-level message (or None).

    `attachment_ids` are uploaded files (see mini.attachments) to send with it;
    raises AttachmentError, saving nothing, if any are missing or already used.
    `voice_note` is (recording's attachment id, duration in seconds) for a voice
    note, whose `content` is then its transcript.
    `sender` is (provider, external id) of the platform account that sent a user
    message, e.g. ("discord", "123456789"); raises UnknownIdentityError, saving
    nothing, if it isn't linked to a person.
    """
    attachment_ids = list(attachment_ids)
    if voice_note is not None and voice_note[0] not in attachment_ids:
        attachment_ids.append(voice_note[0])  # the recording is attached too

    with Session(engine, expire_on_commit=False) as session:
        sender_id = people.resolve(session, *sender) if sender is not None else None
        message = ChatMessage(role=role, content=content, parent_id=parent_id, sender_id=sender_id)
        session.add(message)
        session.flush()  # for the id
        attachments.attach(session, attachment_ids, message.id)
        if voice_note is not None:
            recording_id, duration = voice_note
            session.add(VoiceNote(chat_message_id=message.id, attachment_id=recording_id, duration=duration))
        session.commit()
        session.refresh(message, ["attachments", "voice_note", "sender"])

    for queue in _subscribers:
        queue.put_nowait(message)
    return message


def subscribe() -> asyncio.Queue[ChatMessage]:
    """Get a queue that receives every message sent from now on. Pair with `unsubscribe`."""
    queue: asyncio.Queue[ChatMessage] = asyncio.Queue()
    _subscribers.add(queue)
    return queue


def unsubscribe(queue: asyncio.Queue[ChatMessage]) -> None:
    _subscribers.discard(queue)
