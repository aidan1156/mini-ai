"""The user's chat with the orchestrator, as LLM messages."""

import re
from collections.abc import Collection

from sqlalchemy import Engine, or_, select
from sqlalchemy.orm import Session

from mini.database.models import ChatMessage, WorkerConversation, WorkerMessage
from mini.llm import Message
from mini.orchestrator.context.conversations import describe
from mini.orchestrator.context.utils import CHARS_PER_TOKEN

LEADING_LABEL = re.compile(r"^\s*\[#\d+[^\]]*\]\s*")
# At most this many workers in the note after the history (the most recent ones).
MAX_LINKED_WORKERS = 30
QUOTE_CHARS = 60


def chat_history(
    engine: Engine,
    focus: ChatMessage | None,
    max_tokens: int,
    exclude: Collection[int] = (),
) -> list[Message]:
    """Recent chat as LLM messages, oldest first.

    Includes the most recent top-level messages plus the whole of `focus`'s
    thread (the thread it's in, or the one under it), newest kept first when
    over `max_tokens`. The user's messages are prefixed with their id and
    thread; the orchestrator's aren't, so it doesn't copy the prefixes.

    If any of these messages had workers started for them, a final note lists
    those workers, so the orchestrator can check on work it set going.
    """
    budget = max_tokens * CHARS_PER_TOKEN
    picked: dict[int, ChatMessage] = {}

    def take(messages: Collection[ChatMessage]) -> None:
        nonlocal budget
        for message in messages:
            if message.id in exclude or message.id in picked:
                continue
            budget -= len(message.content)
            if budget < 0:
                return
            picked[message.id] = message

    with Session(engine) as session:
        if focus is not None:
            root_id = focus.parent_id or focus.id
            take(session.scalars(
                select(ChatMessage)
                .where(or_(ChatMessage.id == root_id, ChatMessage.parent_id == root_id))
                .order_by(ChatMessage.id.desc())
            ).all())
        if budget > 0:
            take(session.scalars(
                select(ChatMessage)
                .where(ChatMessage.parent_id.is_(None))
                .order_by(ChatMessage.id.desc())
                .limit(200)
            ).all())

    # Oldest first, with each thread's replies straight after the message that started it.
    ordered = sorted(picked.values(), key=lambda m: (m.parent_id or m.id, m.id))
    messages = [
        Message(role=m.role, content=f"{label(m)} {m.content}" if m.role == "user" else m.content)
        for m in ordered
    ]
    if note := _linked_workers_note(engine, picked):
        messages.append(Message(role="user", content=note))
    return messages


def _linked_workers_note(engine: Engine, messages: dict[int, ChatMessage]) -> str | None:
    """Which workers were started for these chat messages, or None if none were."""
    with Session(engine) as session:
        rows = session.execute(
            select(WorkerMessage.owner_message_id, WorkerConversation)
            .join(WorkerConversation, WorkerConversation.id == WorkerMessage.conversation_id)
            .where(WorkerMessage.owner_message_id.in_(messages))
            .distinct()
            .order_by(WorkerConversation.id.desc())
            .limit(MAX_LINKED_WORKERS)
        ).all()
        if not rows:
            return None
        lines = [
            f"- for {_refer_to(messages[owner_id])}: conversation {describe(c)}"
            for owner_id, c in reversed(rows)
        ]

    return "\n".join([
        "<linked_workers>",
        "Automatic note, not from the user: workers started for messages above "
        "(use read_conversation for details).",
        *lines,
        "</linked_workers>",
    ])


def _refer_to(message: ChatMessage) -> str:
    """How the note names a message: the user's by id, the orchestrator's by quoting it."""
    if message.role == "user":
        return f"#{message.id}"
    quote = message.content if len(message.content) <= QUOTE_CHARS else message.content[:QUOTE_CHARS] + "…"
    return f'your message "{quote}"'


def label(message: ChatMessage) -> str:
    if message.parent_id is None:
        return f"[#{message.id}]"
    return f"[#{message.id} in thread #{message.parent_id}]"


def strip_label(text: str) -> str:
    """Remove a leading [#id] prefix, in case the model copies the format anyway."""
    return LEADING_LABEL.sub("", text, count=1)
