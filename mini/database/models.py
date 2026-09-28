from datetime import datetime

from sqlalchemy import ForeignKey, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# Each table must also be created by a migration in mini/database/migrations/
# -- the models don't create tables themselves.


class WorkerConversation(Base):
    __tablename__ = "worker_conversations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(unique=True)
    description: Mapped[str]
    cwd: Mapped[str]  # Claude Code can only resume a session from the directory it started in
    status: Mapped[str] = mapped_column(server_default="idle")  # 'running' | 'idle' | 'error'
    # The routine run that started this conversation, if any.
    routine_run_id: Mapped[int | None] = mapped_column(ForeignKey("routine_runs.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.current_timestamp())
    last_used: Mapped[datetime] = mapped_column(
        server_default=func.current_timestamp(), onupdate=func.current_timestamp()
    )

    messages: Mapped[list["WorkerMessage"]] = relationship(
        back_populates="conversation",
        order_by="WorkerMessage.id",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    routine_run: Mapped["RoutineRun | None"] = relationship()


class WorkerMessage(Base):
    __tablename__ = "worker_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("worker_conversations.id", ondelete="CASCADE")
    )
    type: Mapped[str]  # 'prompt' | 'assistant' | 'tool_result' | 'system' | 'result'
    text: Mapped[str | None]
    raw: Mapped[str] 
    message_uuid: Mapped[str | None] = mapped_column(unique=True)
    # The chat message this turn is acting on behalf of; replies to the user go in its thread.
    owner_message_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.current_timestamp())

    conversation: Mapped[WorkerConversation] = relationship(back_populates="messages")
    owner_message: Mapped["ChatMessage | None"] = relationship()


class ChatMessage(Base):
    """A message in the user's chat with the orchestrator agent."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    role: Mapped[str]  # 'user' | 'assistant'
    content: Mapped[str]
    # None for a top-level message; otherwise the message that started the thread.
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.current_timestamp())

    parent: Mapped["ChatMessage | None"] = relationship(
        back_populates="replies", remote_side="ChatMessage.id"
    )
    replies: Mapped[list["ChatMessage"]] = relationship(
        back_populates="parent",
        order_by="ChatMessage.id",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class DiscordMessage(Base):
    """Links a chat message to the Discord message showing it (used by mini/discord/)."""

    __tablename__ = "discord_messages"

    chat_message_id: Mapped[int] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="CASCADE"), primary_key=True
    )
    discord_message_id: Mapped[int] = mapped_column(unique=True)
    channel_id: Mapped[int]  # the channel (or thread) the message is in
    thread_id: Mapped[int | None]  # the thread started from this message, once there is one
    created_at: Mapped[datetime] = mapped_column(server_default=func.current_timestamp())


class Routine(Base):
    """Instructions for the orchestrator, run on a timer or when asked (e.g. a morning brief)."""

    __tablename__ = "routines"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str]
    instructions: Mapped[str]  # for the orchestrator, not a worker
    cron: Mapped[str | None]  # 5-field cron in the server's local time; None = only run when asked
    enabled: Mapped[bool] = mapped_column(server_default="1")
    next_run_at: Mapped[datetime | None]  # UTC, like CURRENT_TIMESTAMP; None without a cron
    created_at: Mapped[datetime] = mapped_column(server_default=func.current_timestamp())
    # Routines are soft-deleted, so their runs can always look them up.
    deleted_at: Mapped[datetime | None]


class RoutineRun(Base):
    """One time a routine ran, so the workers it started can be grouped."""

    __tablename__ = "routine_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    routine_id: Mapped[int] = mapped_column(ForeignKey("routines.id"))
    started_at: Mapped[datetime] = mapped_column(server_default=func.current_timestamp())

    routine: Mapped[Routine] = relationship()
