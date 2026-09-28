"""Routines: instructions for the orchestrator that run on a timer or when asked.

The scheduler here only decides *when* a routine runs; the run itself is an
orchestrator turn (see Orchestrator.start_routine).

Cron expressions are in the server's local time; times are stored in UTC.
If the server was down when a run was due, it runs once when it comes back.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from croniter import croniter
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from mini.database.models import Routine

logger = logging.getLogger(__name__)

CHECK_SECONDS = 30


def utc_now() -> datetime:
    """Naive UTC, matching how SQLite stores CURRENT_TIMESTAMP."""
    return datetime.now(UTC).replace(tzinfo=None)


def next_run(cron: str, after: datetime) -> datetime:
    """The next time `cron` (local time) fires after `after` (naive UTC), as naive UTC."""
    after_local = after.replace(tzinfo=UTC).astimezone()
    return croniter(cron, after_local).get_next(datetime).astimezone(UTC).replace(tzinfo=None)


def local_time(utc: datetime) -> str:
    return utc.replace(tzinfo=UTC).astimezone().strftime("%Y-%m-%d %H:%M")


async def run_scheduler(engine: Engine, start: Callable[[int], None]) -> None:
    """Call `start(routine_id)` for each routine as it comes due, forever.

    Run this as a background task.
    """
    while True:
        try:
            _start_due(engine, start)
        except Exception:
            logger.exception("Routine scheduler check failed")
        await asyncio.sleep(CHECK_SECONDS)


def _start_due(engine: Engine, start: Callable[[int], None]) -> None:
    now = utc_now()
    with Session(engine) as session:
        due = session.scalars(
            select(Routine)
            .where(Routine.enabled)
            .where(Routine.next_run_at.is_not(None))
            .where(Routine.next_run_at <= now)
        ).all()
        # Move each on before starting it, so a failing run doesn't retry every check
        # and runs missed while the server was down only happen once.
        for routine in due:
            routine.next_run_at = next_run(routine.cron, now)
        session.commit()
        due_ids = [routine.id for routine in due]

    for routine_id in due_ids:
        start(routine_id)
