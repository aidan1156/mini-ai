"""Worker transcripts, as the orchestrator reads them."""

from collections.abc import Collection

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from mini.database.models import WorkerMessage
from mini.orchestrator.context.utils import CHARS_PER_TOKEN


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
