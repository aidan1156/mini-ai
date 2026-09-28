"""Builds the text the orchestrator sees: chat history, worker transcripts, etc."""

import re
from collections.abc import Collection
from pathlib import Path

from sqlalchemy import Engine, or_, select
from sqlalchemy.orm import Session

from mini.database.models import ChatMessage, WorkerConversation, WorkerMessage
from mini.llm import Message

# Rough conversion for capping text by tokens without a tokenizer.
CHARS_PER_TOKEN = 4

LEADING_LABEL = re.compile(r"^\s*\[#\d+[^\]]*\]\s*")


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
    return [
        Message(role=m.role, content=f"{label(m)} {m.content}" if m.role == "user" else m.content)
        for m in ordered
    ]


def label(message: ChatMessage) -> str:
    if message.parent_id is None:
        return f"[#{message.id}]"
    return f"[#{message.id} in thread #{message.parent_id}]"


def strip_label(text: str) -> str:
    """Remove a leading [#id] prefix, in case the model copies the format anyway."""
    return LEADING_LABEL.sub("", text, count=1)


def projects(projects_dir: Path) -> str:
    """The project folders workers can run in, one per line."""
    names = sorted(
        (p.name for p in projects_dir.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=str.lower,
    )
    return "\n".join(f"- {name}" for name in names) or "(none yet)"


def project_path(projects_dir: Path, project: str) -> Path | None:
    """The folder for `project`, or None unless it's a folder directly inside projects_dir."""
    path = (projects_dir / project).resolve()
    # No "..", absolute paths or nesting.
    if path.parent != projects_dir or not path.is_dir():
        return None
    return path


def recent_conversations(engine: Engine, n: int) -> str:
    """The `n` most recently used worker conversations, one per line."""
    with Session(engine) as session:
        conversations = session.scalars(
            select(WorkerConversation).order_by(WorkerConversation.last_used.desc()).limit(n)
        ).all()
    return "\n".join(describe(c) for c in conversations) or "(none yet)"


def describe(conversation: WorkerConversation) -> str:
    return f'#{conversation.id} [{conversation.status}] "{conversation.description}" (cwd: {conversation.cwd})'


def transcript(
    engine: Engine,
    conversation_id: int,
    max_tokens: int,
    types: Collection[str] = ("prompt", "assistant", "result"),
) -> str:
    """A conversation's text messages (no tool calls), capped to the most recent `max_tokens`."""
    budget = max_tokens * CHARS_PER_TOKEN
    lines: list[str] = []

    with Session(engine) as session:
        messages = session.scalars(
            select(WorkerMessage)
            .where(WorkerMessage.conversation_id == conversation_id)
            .where(WorkerMessage.type.in_(types))
            .where(WorkerMessage.text.is_not(None))
            .order_by(WorkerMessage.id.desc())
        )
        for message in messages:
            line = f"[{message.type}] {message.text}"
            budget -= len(line)
            if budget < 0:
                lines.append("[earlier messages cut]")
                break
            lines.append(line)

    lines.reverse()
    return "\n\n".join(lines) or "(no messages)"


def owned_conversations(engine: Engine, owner_message_id: int) -> list[WorkerConversation]:
    """Conversations with any messages working for this chat message."""
    with Session(engine) as session:
        return list(session.scalars(
            select(WorkerConversation)
            .where(WorkerConversation.id.in_(
                select(WorkerMessage.conversation_id)
                .where(WorkerMessage.owner_message_id == owner_message_id)
            ))
            .order_by(WorkerConversation.id)
        ))


def run_conversations(engine: Engine, routine_run_id: int) -> list[WorkerConversation]:
    """Conversations started by this routine run."""
    with Session(engine) as session:
        return list(session.scalars(
            select(WorkerConversation)
            .where(WorkerConversation.routine_run_id == routine_run_id)
            .order_by(WorkerConversation.id)
        ))
