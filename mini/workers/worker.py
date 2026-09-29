"""Run Claude Code sessions as workers and persist their messages.

Each turn runs `claude -p` as a subprocess with `--output-format stream-json`
in a background task, and every message it streams back is saved to
`worker_messages` as it arrives. The public functions return as soon as the
turn has started -- read the results with `get_recent_messages`, and check
`WorkerConversation.status` to see whether a turn is still running.

Each turn can have an owner: the chat message it is acting on behalf of.
Every message saved during the turn records it as `owner_message_id`. Pass
`on_turn_end` to be called with the conversation id once a turn has finished
(successfully or not) -- the orchestrator uses this to react to results.
"""

import asyncio
import json
import logging
import shutil
import uuid
from collections.abc import Awaitable, Callable, Collection
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, select, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from mini.attachments import ATTACHMENTS_DIR
from mini.database.models import WorkerConversation, WorkerMessage
from mini.mcp_servers import worker_mcp_args

logger = logging.getLogger(__name__)

# A single stream line can hold a whole tool result (e.g. a file's contents).
STREAM_LINE_LIMIT = 64 * 1024 * 1024

# asyncio only keeps weak references to tasks, so hold on to running turns
# here or they can be garbage collected mid-turn.
_running_turns: set[asyncio.Task] = set()

# Called with the conversation id after a turn finishes and its status is saved.
TurnEndHandler = Callable[[int], Awaitable[None]]


class WorkerError(Exception):
    pass


class WorkerBusyError(WorkerError):
    """The conversation is already running a turn."""


async def create_conversation(
    engine: Engine,
    prompt: str,
    description: str,
    cwd: Path,
    model: str | None = None,
    owner_message_id: int | None = None,
    on_turn_end: TurnEndHandler | None = None,
    routine_run_id: int | None = None,
) -> int:
    """Start a new Claude Code session with `prompt`.

    Returns the internal conversation id immediately; the turn runs in the background.
    """
    claude = _find_claude()
    session_id = str(uuid.uuid4())
    cwd = Path(cwd).resolve()

    with Session(engine) as session:
        conversation = WorkerConversation(
            session_id=session_id,
            description=description,
            cwd=str(cwd),
            status="running",
            routine_run_id=routine_run_id,
        )
        session.add(conversation)
        session.commit()
        conversation_id = conversation.id

    _start_turn(
        engine, conversation_id, prompt, cwd,
        claude, ["--session-id", session_id], model,
        owner_message_id, on_turn_end,
    )
    return conversation_id


async def send_message(
    engine: Engine,
    conversation_id: int,
    prompt: str,
    model: str | None = None,
    owner_message_id: int | None = None,
    on_turn_end: TurnEndHandler | None = None,
) -> None:
    """Send `prompt` to an existing conversation.

    Returns immediately; the turn runs in the background. Raises
    WorkerBusyError if the conversation is still running its previous turn.
    """
    claude = _find_claude()

    with Session(engine) as session:
        # Claim the conversation atomically so two turns can't run at once.
        claimed = session.execute(
            update(WorkerConversation)
            .where(WorkerConversation.id == conversation_id)
            .where(WorkerConversation.status != "running")
            .values(status="running")
        ).rowcount
        session.commit()

        conversation = session.get(WorkerConversation, conversation_id)
        if conversation is None:
            raise WorkerError(f"No conversation with id {conversation_id}")
        if not claimed:
            raise WorkerBusyError(f"Conversation {conversation_id} is already running")

        session_id, cwd = conversation.session_id, Path(conversation.cwd)

    _start_turn(
        engine, conversation_id, prompt, cwd,
        claude, ["--resume", session_id], model,
        owner_message_id, on_turn_end,
    )


def get_recent_messages(
    engine: Engine,
    conversation_id: int,
    n: int,
    types: Collection[str] | None = None,
) -> list[WorkerMessage]:
    """Return the most recent `n` messages, oldest first.

    Pass `types` (e.g. {"prompt", "assistant"}) to only include those types.
    """
    query = (
        select(WorkerMessage)
        .where(WorkerMessage.conversation_id == conversation_id)
        .order_by(WorkerMessage.id.desc())
        .limit(n)
    )
    if types is not None:
        query = query.where(WorkerMessage.type.in_(types))

    with Session(engine) as session:
        messages = list(session.scalars(query))

    messages.reverse()
    return messages


def mark_interrupted(engine: Engine) -> int:
    """Mark conversations left 'running' by an earlier process as 'error'.

    Call once at startup: a turn can't survive a restart, and a conversation
    stuck in 'running' could never be sent another message. Returns how many
    were marked.
    """
    with Session(engine) as session:
        marked = session.execute(
            update(WorkerConversation)
            .where(WorkerConversation.status == "running")
            .values(status="error")
        ).rowcount
        session.commit()
    if marked:
        logger.warning("Marked %s conversation(s) interrupted by a restart as 'error'", marked)
    return marked


def _find_claude() -> str:
    claude = shutil.which("claude")
    if claude is None:
        raise WorkerError("Could not find the `claude` CLI on PATH")
    return claude


def _start_turn(
    engine: Engine,
    conversation_id: int,
    prompt: str,
    cwd: Path,
    claude: str,
    session_args: list[str],
    model: str | None,
    owner_message_id: int | None,
    on_turn_end: TurnEndHandler | None,
) -> None:
    # Claude Code doesn't echo the prompt back in the stream, so save it
    # ourselves -- before returning, so it's readable straight away.
    _save_message(
        engine, conversation_id, "prompt", prompt,
        {"type": "user", "content": prompt}, None, owner_message_id,
    )

    # Let workers read the chat's attachments (the orchestrator hands them over by path).
    ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
    args = [
        claude, "-p", "--output-format", "stream-json", "--verbose",
        "--permission-mode", "auto", "--add-dir", str(ATTACHMENTS_DIR),
        *worker_mcp_args(), *session_args,
    ]
    if model:
        args += ["--model", model]

    task = asyncio.create_task(
        _run_turn(engine, conversation_id, prompt, cwd, args, owner_message_id, on_turn_end)
    )
    _running_turns.add(task)
    task.add_done_callback(_running_turns.discard)


async def _run_turn(
    engine: Engine,
    conversation_id: int,
    prompt: str,
    cwd: Path,
    args: list[str],
    owner_message_id: int | None,
    on_turn_end: TurnEndHandler | None,
) -> None:
    status = "error"
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=STREAM_LINE_LIMIT,
        )
        # Read stderr alongside stdout so a full stderr pipe can't stall the process.
        stderr_task = asyncio.create_task(process.stderr.read())

        # Pass the prompt on stdin to avoid Windows command-line quoting and length limits.
        process.stdin.write(prompt.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()

        result: dict[str, Any] | None = None
        async for line in process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue

            parsed = _parse_stream_message(message)
            if parsed is not None:
                message_type, text = parsed
                _save_message(
                    engine, conversation_id, message_type, text,
                    message, message.get("uuid"), owner_message_id,
                )

            if message.get("type") == "result":
                result = message

        return_code = await process.wait()
        stderr = (await stderr_task).decode("utf-8", errors="replace").strip()

        if return_code != 0 or result is None or result.get("is_error"):
            detail = stderr or (result or {}).get("result") or f"exit code {return_code}"
            logger.error("Conversation %s turn failed: %s", conversation_id, detail)
        else:
            status = "idle"
    except Exception:
        # Nobody awaits this task, so log rather than raise.
        logger.exception("Conversation %s turn crashed", conversation_id)
    finally:
        if process is not None and process.returncode is None:
            process.kill()  # e.g. the task was cancelled mid-turn
        _set_status(engine, conversation_id, status)
        if on_turn_end is not None:
            await _notify_turn_end(on_turn_end, conversation_id)


async def _notify_turn_end(on_turn_end: TurnEndHandler, conversation_id: int) -> None:
    try:
        await on_turn_end(conversation_id)
    except Exception:
        # Nobody awaits the turn's task, so log rather than raise.
        logger.exception("on_turn_end handler failed for conversation %s", conversation_id)


def _parse_stream_message(message: dict[str, Any]) -> tuple[str, str | None] | None:
    """Map a stream-json message to (type, text), or None if it isn't worth storing."""
    match message.get("type"):
        case "assistant":
            return "assistant", _join_text(message["message"]["content"])
        case "user":
            content = message["message"]["content"]
            if isinstance(content, list) and any(b.get("type") == "tool_result" for b in content):
                texts = [_join_text(b.get("content")) for b in content if b.get("type") == "tool_result"]
                return "tool_result", "\n".join(t for t in texts if t) or None
            return "prompt", _join_text(content)
        case "system" if message.get("subtype") == "thinking_tokens":
            return None  # progress updates while Claude is thinking
        case "system":
            return "system", message.get("subtype")
        case "result":
            return "result", message.get("result")
        case _:
            return None  # rate_limit_event, stream_event, etc.


def _join_text(content: Any) -> str | None:
    """Pull the plain text out of a message's content (a string or a list of blocks)."""
    if isinstance(content, str):
        return content or None
    if not isinstance(content, list):
        return None
    texts = [b["text"] for b in content if b.get("type") == "text" and b.get("text")]
    return "\n".join(texts) or None


def _save_message(
    engine: Engine,
    conversation_id: int,
    message_type: str,
    text: str | None,
    raw: dict[str, Any],
    message_uuid: str | None,
    owner_message_id: int | None,
) -> None:
    with Session(engine) as session:
        session.execute(
            insert(WorkerMessage)
            .values(
                conversation_id=conversation_id,
                type=message_type,
                text=text,
                raw=json.dumps(raw),
                message_uuid=message_uuid,
                owner_message_id=owner_message_id,
            )
            .on_conflict_do_nothing()  # already saved (same message_uuid)
        )
        session.commit()


def _set_status(engine: Engine, conversation_id: int, status: str) -> None:
    with Session(engine) as session:
        session.execute(
            update(WorkerConversation)
            .where(WorkerConversation.id == conversation_id)
            .values(status=status)
        )
        session.commit()
