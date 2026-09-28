"""Where the orchestrator's messages about worker updates go in the user's chat.

Each worker turn has an owner: the chat message it's working for, stored on
its messages as `WorkerMessage.owner_message_id`. The rules:

- A turn started without an owner inherits the conversation's latest owner,
  so a conversation's updates keep landing in one thread.
- Owner is top-level: the orchestrator chooses between replying in its
  thread and posting top-level (e.g. for a quick check).
- Owner is itself a reply: the message always goes in that thread.
- No owner: the message is posted top-level and becomes the owner of the
  conversation's messages (see `adopt`), so later updates go in its thread.
"""

from collections.abc import Collection

from sqlalchemy import Engine, select, update
from sqlalchemy.orm import Session

from mini.database.models import ChatMessage, WorkerMessage


def resolve_owner(engine: Engine, conversation_id: int, owner_message_id: int | None = None) -> int | None:
    """The owner for a turn: the one given, else the conversation's latest owner."""
    if owner_message_id is not None:
        return owner_message_id

    with Session(engine) as session:
        return session.scalar(
            select(WorkerMessage.owner_message_id)
            .where(WorkerMessage.conversation_id == conversation_id)
            .where(WorkerMessage.owner_message_id.is_not(None))
            .order_by(WorkerMessage.id.desc())
            .limit(1)
        )


def can_choose_thread(owner: ChatMessage | None) -> bool:
    """Whether the orchestrator gets to pick thread vs top-level for its message."""
    return owner is not None and owner.parent_id is None


def reply_parent(owner: ChatMessage | None, in_thread: bool) -> int | None:
    """The parent_id for a message about the owner's work.

    `in_thread` only counts when the owner is top-level (see can_choose_thread).
    """
    if owner is None:
        return None
    if owner.parent_id is not None:
        return owner.parent_id  # owner is in a thread: always reply there
    return owner.id if in_thread else None


def adopt(engine: Engine, owner_message_id: int, conversation_ids: Collection[int]) -> None:
    """Make `owner_message_id` the owner of these conversations' ownerless messages."""
    if not conversation_ids:
        return
    with Session(engine) as session:
        session.execute(
            update(WorkerMessage)
            .where(WorkerMessage.conversation_id.in_(conversation_ids))
            .where(WorkerMessage.owner_message_id.is_(None))
            .values(owner_message_id=owner_message_id)
        )
        session.commit()
