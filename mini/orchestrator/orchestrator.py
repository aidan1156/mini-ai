"""The orchestrator agent: the one agent the user chats with.

It runs a turn in three cases:

- `handle_user_message`: the user said something. For a top-level message it
  chooses whether to reply inline or in a thread under it.
- `on_worker_turn_end`: a worker session finished a turn. The orchestrator is
  shown the result and decides whether to tell the user. Its message goes
  wherever the conversation about that work already is (see
  mini.orchestrator.owners).
- `start_routine`: a routine is due (or was asked for). The orchestrator gets
  its instructions in a hidden message -- nothing is posted to the chat -- and
  the workers it starts are grouped as one run, so it can wait for all of
  them before telling the user.

Either way it talks to the user with its send_chat_message tool.

Turns run one at a time. Worker updates that arrive while a turn is running
are batched, one orchestrator turn per owner message (or per routine run).
"""

import asyncio
import datetime
import logging
from collections import defaultdict
from collections.abc import Collection
from pathlib import Path

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from mini.database.models import ChatMessage, Routine, RoutineRun, WorkerConversation
from mini.llm import LLM, Message
from mini.orchestrator import chat, context
from mini.orchestrator.owners import adopt, resolve_owner, update_parent
from mini.orchestrator.tools import ToolRunner, run_tool

logger = logging.getLogger(__name__)

HISTORY_TOKENS = 8000
UPDATE_TRANSCRIPT_TOKENS = 3000
RECENT_CONVERSATIONS = 10
MAX_STEPS = 20  # LLM calls per orchestrator turn

SYSTEM_PROMPT = """\
You are the user's assistant called Mini, you are here for managing their AI agent sessions ("workers").
The user chats with you instead of with each session. You can start sessions,
send them follow-ups, read what they've done, and keep their descriptions accurate.
The time is currently {now}.

- Workers run in the background. After starting one or sending it a message, tell
  the user what you've set going; you'll get a <worker_update> when it finishes.
- Answer questions about ongoing work by reading the relevant conversation.
- Give workers self-contained prompts: they can't see this chat.
- The user only sees what you send with send_chat_message.
- For a top-level message, choose in_thread: false for quick answers and small
  tasks, true for bigger work with several updates to come. Updates about the
  work later go wherever your first reply went.
- Routines are instructions you've saved for yourself, run on a timer or when the
  user asks. When one runs you get a <routine> message with its instructions.
- The user's messages are prefixed with [#id] or [#id in thread #root]. Don't
  write these prefixes yourself.
- Be brief.

Projects you can start workers in (folders in {projects_dir}):
{projects}

Recently used worker conversations:
{conversations}
"""


class Orchestrator:
    def __init__(self, engine: Engine, llm: LLM, projects_dir: Path):
        self.engine = engine
        self.llm = llm
        self.projects_dir = projects_dir
        self._lock = asyncio.Lock()
        self._pending: set[int] = set()  # conversations with unhandled updates
        self._drain_task: asyncio.Task | None = None
        # asyncio only keeps weak references to tasks, so hold on to turns in progress.
        self._tasks: set[asyncio.Task] = set()

    def handle_user_message(self, content: str, parent_id: int | None = None) -> ChatMessage:
        """Save the user's message and return it; the orchestrator replies in the background."""
        user_message = chat.send_message(self.engine, "user", content, parent_id)
        self._in_background(self._reply(user_message))
        return user_message

    def start_routine(self, routine_id: int, owner_message_id: int | None = None) -> None:
        """Run a routine in the background, once any current turn is done.

        `owner_message_id` is the chat message that asked for it, if any.
        """
        self._in_background(self._run_routine(routine_id, owner_message_id))

    async def on_worker_turn_end(self, conversation_id: int) -> None:
        """Pass to workers as `on_turn_end`. Queues the update and returns straight away."""
        self._pending.add(conversation_id)
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.create_task(self._drain())

    def _in_background(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Orchestrator turn failed", exc_info=task.exception())

    def _runner(self, **kwargs) -> ToolRunner:
        return ToolRunner(
            self.engine, self.on_worker_turn_end, self.start_routine, self.projects_dir, **kwargs
        )

    async def _reply(self, user_message: ChatMessage) -> None:
        async with self._lock:
            messages = context.chat_history(self.engine, user_message, HISTORY_TOKENS, exclude={user_message.id})
            messages.append(Message(role="user", content=f"{context.label(user_message)} {user_message.content}"))

            # Reply next to the user's message, or (if it's top-level) optionally in a thread under it.
            runner = self._runner(
                owner_id=user_message.id,
                parent_id=user_message.parent_id,
                thread_under=user_message.id if user_message.parent_id is None else None,
            )
            final_text = await self._run(messages, runner)

            # If it answered in plain text instead of using the tool, send that.
            if final_text and not runner.posted:
                await runner.send_chat_message(final_text)

    async def _run_routine(self, routine_id: int, owner_message_id: int | None) -> None:
        async with self._lock:
            with Session(self.engine, expire_on_commit=False) as session:
                routine = session.get(Routine, routine_id)
                if routine is None:
                    return  # deleted since it was queued
                run = RoutineRun(routine_id=routine.id, routine_name=routine.name)
                session.add(run)
                session.commit()
                owner = session.get(ChatMessage, owner_message_id) if owner_message_id is not None else None

            messages = context.chat_history(self.engine, owner, HISTORY_TOKENS)
            messages.append(Message(role="user", content=self._routine_text(routine, run, owner)))

            runner = self._runner(
                owner_id=owner_message_id,
                parent_id=update_parent(self.engine, owner),
                routine_run_id=run.id,
            )
            await self._run(messages, runner)  # plain final text is ignored: saying nothing is allowed

            if owner is None and runner.posted:
                adopt(self.engine, runner.posted[0].id, runner.touched)

    def _routine_text(self, routine: Routine, run: RoutineRun, owner: ChatMessage | None) -> str:
        if owner is not None:
            trigger = f"The user asked for it in chat message {context.label(owner)}."
        elif routine.cron:
            trigger = f"It runs on a timer (cron {routine.cron!r})."
        else:
            trigger = "It was started automatically."

        return "\n".join([
            "<routine>",
            "This is an automatic trigger, not a message from the user.",
            f'Run your routine #{routine.id} "{routine.name}" (this is run #{run.id}). {trigger}',
            "Its instructions:",
            "",
            routine.instructions,
            "",
            "Workers you start now are part of this run: each one's update lists the others, so you "
            "can wait for all of them and send the user one combined message. Only message the user "
            "now if there's already something to say (e.g. nothing needed doing).",
            self._where_messages_go(owner),
            "</routine>",
        ])

    async def _drain(self) -> None:
        while self._pending:
            async with self._lock:
                batch, self._pending = self._pending, set()

                # Group by owner message; work nobody owns yet is grouped by its routine run.
                groups: dict[tuple[int | None, int | None], list[int]] = defaultdict(list)
                with Session(self.engine) as session:
                    for conversation_id in sorted(batch):
                        owner_id = resolve_owner(self.engine, conversation_id)
                        run_id = None
                        if owner_id is None:
                            run_id = session.get(WorkerConversation, conversation_id).routine_run_id
                        groups[owner_id, run_id].append(conversation_id)

                for (owner_id, run_id), conversation_ids in groups.items():
                    try:
                        await self._handle_update(owner_id, run_id, conversation_ids)
                    except Exception:
                        # Nobody awaits this task, so log and carry on with the rest.
                        logger.exception("Orchestrator failed on update for conversations %s", conversation_ids)

    async def _handle_update(self, owner_id: int | None, run_id: int | None, conversation_ids: list[int]) -> None:
        with Session(self.engine, expire_on_commit=False) as session:
            owner = session.get(ChatMessage, owner_id) if owner_id is not None else None
            run_ids = {session.get(WorkerConversation, i).routine_run_id for i in conversation_ids}

        messages = context.chat_history(self.engine, owner, HISTORY_TOKENS)
        messages.append(Message(role="user", content=self._update_text(owner, run_id, conversation_ids)))

        runner = self._runner(
            owner_id=owner_id,
            parent_id=update_parent(self.engine, owner),
            # Follow-up workers join the run these updates came from.
            routine_run_id=next(iter(run_ids)) if len(run_ids) == 1 else None,
        )
        await self._run(messages, runner)  # plain final text is ignored: saying nothing is allowed

        if owner is None and runner.posted:
            adopted = {*conversation_ids, *runner.touched}
            if run_id is not None:
                # The whole run shares the thread, including workers still going.
                adopted |= {c.id for c in context.run_conversations(self.engine, run_id)}
            adopt(self.engine, runner.posted[0].id, adopted)

    def _update_text(self, owner: ChatMessage | None, run_id: int | None, conversation_ids: Collection[int]) -> str:
        with Session(self.engine) as session:
            conversations = [session.get(WorkerConversation, i) for i in conversation_ids]
            runs = {c.id: c.routine_run for c in conversations}

        parts = ["<worker_update>", "This is an automatic update, not a message from the user.", ""]
        for conversation in conversations:
            parts.append(f"Conversation {context.describe(conversation)} finished a turn.")
            if (run := runs[conversation.id]) is not None:
                parts.append(f'It was started by run #{run.id} of your routine "{run.routine_name}".')
            parts += [
                "Transcript (prompts sent and results):",
                context.transcript(self.engine, conversation.id, UPDATE_TRANSCRIPT_TOKENS, types=("prompt", "result")),
                "",
            ]

        if owner is not None:
            parts.append(f"This work was asked for in chat message {context.label(owner)}.")
            others = [c for c in context.owned_conversations(self.engine, owner.id) if c.id not in conversation_ids]
            if others:
                parts.append("Other conversations working for it:")
                parts += [f"- {context.describe(c)}" for c in others]
        elif run_id is not None:
            others = [c for c in context.run_conversations(self.engine, run_id) if c.id not in conversation_ids]
            if others:
                parts.append(f"Other conversations in routine run #{run_id}:")
                parts += [f"- {context.describe(c)}" for c in others]
            parts.append("Nobody is waiting on this in the chat.")
        else:
            parts.append("Nobody is waiting on this in the chat.")

        parts.append(self._where_messages_go(owner))
        parts.append(
            "If nothing needs saying yet (e.g. other conversations are still working on the "
            "same thing), end your turn without sending anything. You can also react by "
            "sending workers follow-ups."
        )
        parts.append("</worker_update>")
        return "\n".join(parts)

    def _where_messages_go(self, owner: ChatMessage | None) -> str:
        if owner is None:
            return (
                "If you send a message it will be posted top-level, and later updates about this "
                "work will go in its thread."
            )
        parent_id = update_parent(self.engine, owner)
        where = f"in thread #{parent_id}" if parent_id is not None else "top-level, like your earlier reply"
        return f"Messages you send with send_chat_message will go {where}."

    async def _run(self, messages: list[Message], runner: ToolRunner) -> str | None:
        """The tool-use loop. Returns the model's final text."""
        system = SYSTEM_PROMPT.format(
            now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            projects_dir=self.projects_dir,
            projects=context.projects(self.projects_dir),
            conversations=context.recent_conversations(self.engine, RECENT_CONVERSATIONS),
        )
        tools = runner.tools()

        for _ in range(MAX_STEPS):
            reply = await self.llm.complete(system, messages, tools)
            messages.append(reply)
            if not reply.tool_calls:
                return reply.content
            for call in reply.tool_calls:
                result = await run_tool(tools, call)
                messages.append(Message(role="tool", content=result, tool_call_id=call.id))

        logger.warning("Orchestrator turn hit the %s step limit", MAX_STEPS)
        return None
