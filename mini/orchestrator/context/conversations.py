"""Finding and describing worker conversations."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from mini.database.models import WorkerConversation, WorkerMessage

# The system prompt lists conversations used in the last RECENT_HOURS, at most RECENT_LIMIT of them.
RECENT_HOURS = 30
RECENT_LIMIT = 20


def describe(conversation: WorkerConversation) -> str:
    return f'#{conversation.id} [{conversation.status}] "{conversation.description}" (cwd: {conversation.cwd})'


def recent_conversations(engine: Engine) -> str:
    """Worker conversations used in the last RECENT_HOURS (newest RECENT_LIMIT), one per line."""
    # last_used is naive UTC, like CURRENT_TIMESTAMP.
    since = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=RECENT_HOURS)
    with Session(engine) as session:
        conversations = session.scalars(
            select(WorkerConversation)
            .where(WorkerConversation.last_used >= since)
            .order_by(WorkerConversation.last_used.desc())
            .limit(RECENT_LIMIT)
        ).all()
    return "\n".join(describe(c) for c in conversations) or f"(none in the last {RECENT_HOURS} hours)"


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
