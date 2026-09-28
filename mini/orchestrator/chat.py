"""The user's chat with the orchestrator."""

from typing import Literal

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from mini.database.models import ChatMessage


def send_message(
    engine: Engine,
    role: Literal["user", "assistant"],
    content: str,
    parent_id: int | None = None,
) -> ChatMessage:
    """Add a message to the chat. `parent_id` must be a top-level message (or None)."""
    with Session(engine, expire_on_commit=False) as session:
        message = ChatMessage(role=role, content=content, parent_id=parent_id)
        session.add(message)
        session.commit()
    return message
