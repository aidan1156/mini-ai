"""The user's chat with the orchestrator.

Anything that shows the chat (the API's event stream, for one) can `subscribe`
to get every new message as it's saved.
"""

import asyncio
from collections.abc import Collection
from typing import Literal

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from mini import attachments
from mini.database.models import ChatMessage

_subscribers: set[asyncio.Queue[ChatMessage]] = set()


def send_message(
    engine: Engine,
    role: Literal["user", "assistant"],
    content: str,
    parent_id: int | None = None,
    attachment_ids: Collection[int] = (),
) -> ChatMessage:
    """Add a message to the chat. `parent_id` must be a top-level message (or None).

    `attachment_ids` are uploaded files (see mini.attachments) to send with it;
    raises AttachmentError, saving nothing, if any are missing or already used.
    """
    with Session(engine, expire_on_commit=False) as session:
        message = ChatMessage(role=role, content=content, parent_id=parent_id)
        session.add(message)
        session.flush()  # for the id
        attachments.attach(session, attachment_ids, message.id)
        session.commit()
        session.refresh(message, ["attachments"])

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
