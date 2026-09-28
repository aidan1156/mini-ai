"""Where the orchestrator's messages about worker updates go in the user's chat.

Each worker turn has an owner: the chat message it's working for, stored on
its messages as `WorkerMessage.owner_message_id`. A turn started without an
owner inherits the conversation's latest owner, so a conversation's updates
keep landing in the same place.

Where to reply is decided once per owner, when the orchestrator first replies
to the user's message: in a thread under it, or top-level for small things.
Updates then follow the conversation (see `update_parent`):

- Owner is itself a reply: always that thread.
- There's a thread under the owner, or the owner is one of the orchestrator's
  own messages: that thread.
- Otherwise (the orchestrator replied top-level): top-level.
- No owner: top-level, and the message becomes the owner of the
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


def update_parent(engine: Engine, owner: ChatMessage | None) -> int | None:
    """The parent_id for an update about the owner's work."""
    if owner is None:
        return None
    if owner.parent_id is not None:
        return owner.parent_id
    if owner.role == "assistant":
        return owner.id  # the orchestrator's own message, adopted as the owner

    with Session(engine) as session:
        has_thread = session.scalar(
            select(ChatMessage.id).where(ChatMessage.parent_id == owner.id).limit(1)
        ) is not None
    return owner.id if has_thread else None


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
