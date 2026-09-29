"""The user's chat with the orchestrator, as LLM messages."""

import re
from collections.abc import Collection

from sqlalchemy import Engine, or_, select
from sqlalchemy.orm import Session

from mini import attachments
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
    threads: Collection[int] = (),
) -> list[Message]:
    """Recent chat as LLM messages, oldest first.

    Includes the most recent top-level messages plus the whole of `focus`'s
    thread (the thread it's in, or the one under it) and of the threads under
    the messages in `threads`, newest kept first when over `max_tokens`. The
    user's messages are prefixed with their id and thread; the orchestrator's
    aren't, so it doesn't copy the prefixes.

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

    roots = [focus.parent_id or focus.id] if focus is not None else []
    roots += [root_id for root_id in threads if root_id not in roots]

    with Session(engine) as session:
        for root_id in roots:
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
    messages = [render(m) for m in ordered]
    if note := _linked_workers_note(engine, picked):
        messages.append(Message(role="user", content=note))
    return messages


def render(message: ChatMessage) -> Message:
    """A chat message as the orchestrator sees it, with any attachments listed.

    The user's messages get a [#id] prefix and their attachments' paths (so they
    can be handed to workers); the orchestrator's own just name what it sent.
    """
    if message.role == "user":
        lines = [f"{label(message)} {message.content}".rstrip()]
        lines += [f"[attached: {attachments.describe(a)}]" for a in message.attachments]
    else:
        lines = [message.content]
        if message.attachments:
            lines.append(f"(sent with: {', '.join(a.filename for a in message.attachments)})")
    return Message(role=message.role, content="\n".join(lines))


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


def search_chat(engine: Engine, text: str, limit: int) -> list[ChatMessage]:
    """The `limit` most recent chat messages containing `text` (ignoring case), newest first."""
    with Session(engine) as session:
        return list(session.scalars(
            select(ChatMessage)
            .where(ChatMessage.content.icontains(text, autoescape=True))
            .order_by(ChatMessage.id.desc())
            .limit(limit)
        ).all())


def describe_match(message: ChatMessage) -> str:
    """A search result: where the message is, who sent it and what it says."""
    who = "user" if message.role == "user" else "you"
    return f"{label(message)} {who}: {message.content}"


def thread_has_replies(engine: Engine, root_id: int) -> bool:
    with Session(engine) as session:
        return session.scalar(select(ChatMessage.id).where(ChatMessage.parent_id == root_id).limit(1)) is not None


def describe_thread(engine: Engine, root_id: int) -> str:
    """One line about the thread under `root_id`: what it's under and how it ends."""
    with Session(engine) as session:
        root = session.get(ChatMessage, root_id)
        replies = session.scalars(
            select(ChatMessage).where(ChatMessage.parent_id == root_id).order_by(ChatMessage.id)
        ).all()
        where = f"the thread under {_refer_to(root)}"
        if not replies:
            return f"{where} (no replies yet; this starts it)"
        last = replies[-1]
        who = "the user" if last.role == "user" else "you"
        return f"{where} ({len(replies)} replies, the latest from {who}: {_quote(last.content)})"


def _refer_to(message: ChatMessage) -> str:
    """How the note names a message: the user's by id, the orchestrator's by quoting it."""
    if message.role == "user":
        return f"#{message.id}"
    return f"your message {_quote(message.content)}"


def _quote(text: str) -> str:
    text = " ".join(text.split())
    return f'"{text}"' if len(text) <= QUOTE_CHARS else f'"{text[:QUOTE_CHARS]}…"'


def label(message: ChatMessage) -> str:
    if message.parent_id is None:
        return f"[#{message.id}]"
    return f"[#{message.id} in thread #{message.parent_id}]"


def strip_label(text: str) -> str:
    """Remove a leading [#id] prefix, in case the model copies the format anyway."""
    return LEADING_LABEL.sub("", text, count=1)
