"""Where the orchestrator's messages go in the user's chat.

Each worker turn has an owner: the chat message it's working for, stored on
its messages as `WorkerMessage.owner_message_id`. A turn started without an
owner inherits the conversation's latest owner, so a conversation's updates
keep landing in the same place.

Where a message goes is either forced or picked by the model from a short list
(see `Placement`):

- Replying to the user: in the same thread if their message is in one;
  otherwise the model picks inline (top-level) or a thread under their message.
- An update about some work (or a routine run) whose owner is in a thread:
  always that thread.
- ...whose owner is top-level: the model picks top-level or one of the threads
  that could be about the work: the one under the owner, or any started since
  the owner was sent (e.g. one the user started off the orchestrator's reply).
- ...with no owner: top-level, and the message becomes the owner of the work
  (see `adopt`), so later updates go in its thread.
"""

from collections.abc import Collection
from dataclasses import dataclass, field

from sqlalchemy import Engine, func, select, update
from sqlalchemy.orm import Session

from mini.database.models import ChatMessage, WorkerMessage

# At most this many threads to choose from, besides the one under the owner.
MAX_THREAD_CHOICES = 5


@dataclass
class Placement:
    """Where the orchestrator's next message can go.

    With no `choices` it goes to `parent_id` (None = top-level). Otherwise the
    model picks: top-level, or the thread under one of the `choices`.
    """

    parent_id: int | None = None
    choices: list[int] = field(default_factory=list)


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


def reply_placement(user_message: ChatMessage) -> Placement:
    """Where to reply to the user's message."""
    if user_message.parent_id is not None:
        return Placement(parent_id=user_message.parent_id)
    return Placement(choices=[user_message.id])


def update_placement(engine: Engine, owner: ChatMessage | None) -> Placement:
    """Where to post about the owner's work."""
    if owner is None:
        return Placement()
    if owner.parent_id is not None:
        return Placement(parent_id=owner.parent_id)

    # Threads under the owner or under anything sent since, most recently active first.
    with Session(engine) as session:
        threads = list(session.scalars(
            select(ChatMessage.parent_id)
            .where(ChatMessage.parent_id >= owner.id)
            .group_by(ChatMessage.parent_id)
            .order_by(func.max(ChatMessage.id).desc())
            .limit(MAX_THREAD_CHOICES)
        ))
    if owner.id not in threads:
        threads.append(owner.id)  # starting a thread under the owner is always an option
    return Placement(choices=threads)


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
