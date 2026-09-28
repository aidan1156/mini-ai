"""The orchestrator's tools, and running them for one orchestrator turn."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from croniter import croniter
from sqlalchemy import Engine, select, update
from sqlalchemy.orm import Session

from mini.database.models import ChatMessage, Routine, WorkerConversation
from mini.llm import Tool, ToolCall
from mini.orchestrator import chat, context, routines
from mini.orchestrator.owners import Placement, resolve_owner
from mini.workers import worker
from mini.workers.worker import TurnEndHandler, WorkerBusyError, WorkerError

logger = logging.getLogger(__name__)

READ_TOKENS = 4000
SEARCH_LIMIT = 20

async def run_tool(tools: list[Tool], call: ToolCall) -> str:
    """Run the tool the model asked for, returning its result (or error) for the model."""
    tool = next((t for t in tools if t.name == call.name), None)
    if tool is None:
        return f"Error: unknown tool {call.name}"
    try:
        return await tool.handler(**call.arguments)
    except (TypeError, WorkerError) as e:
        return f"Error: {e}"  # e.g. missing arguments, or no such conversation
    except Exception as e:
        logger.exception("Tool %s failed", call.name)
        return f"Error: {e}"


class ToolRunner:
    """The tools for one orchestrator turn, and the state they share.

    `owner_id` is the chat message that worker prompts sent this turn work for.
    New workers run in a repo directly inside `projects_dir`.
    `placement` says where send_chat_message posts, or which threads the model
    can pick from. Once it has posted, later messages this turn go in the same place.
    Workers started this turn are part of `routine_run_id`, if set.
    `start_routine(routine_id, owner_message_id)` queues a routine run.
    """

    def __init__(
        self,
        engine: Engine,
        on_turn_end: TurnEndHandler,
        start_routine: Callable[[int, int | None], None],
        projects_dir: Path,
        owner_id: int | None,
        placement: Placement,
        routine_run_id: int | None = None,
    ):
        self.engine = engine
        self.on_turn_end = on_turn_end
        self.start_routine = start_routine
        self.projects_dir = projects_dir
        self.owner_id = owner_id
        self.placement = placement
        self.routine_run_id = routine_run_id
        self.touched: set[int] = set()  # conversations started or messaged this turn
        self.posted: list[ChatMessage] = []

    def tools(self) -> list[Tool]:
        return [
            Tool(
                name="start_conversation",
                description=(
                    "Start a new Claude Code session to work on a task. It runs in the background; "
                    "you'll get an update when it finishes its turn."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "The task, with all the context the session needs."},
                        "description": {"type": "string", "description": "A short summary of the work, used to find it later."},
                        "project": {
                            "type": ["string", "null"],
                            "description": (
                                "The project (folder name) to work in, or null to start in the projects "
                                "folder itself: for general tasks, or when you're not sure which project "
                                "or whether it's possible."
                            ),
                        },
                    },
                    "required": ["prompt", "description", "project"],
                },
                handler=self._start_conversation,
            ),
            Tool(
                name="send_to_conversation",
                description=(
                    "Send a message to an existing Claude Code session, e.g. a follow-up task or an answer "
                    "to its question. Fails if it's still running its last turn."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "conversation_id": {"type": "integer"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["conversation_id", "prompt"],
                },
                handler=self._send_to_conversation,
            ),
            Tool(
                name="read_conversation",
                description="Read a session's recent messages: the prompts it was sent and its text replies.",
                parameters={
                    "type": "object",
                    "properties": {"conversation_id": {"type": "integer"}},
                    "required": ["conversation_id"],
                },
                handler=self._read_conversation,
            ),
            Tool(
                name="rename_conversation",
                description="Change a session's description, e.g. when its work changes. Search relies on these.",
                parameters={
                    "type": "object",
                    "properties": {
                        "conversation_id": {"type": "integer"},
                        "description": {"type": "string"},
                    },
                    "required": ["conversation_id", "description"],
                },
                handler=self._rename_conversation,
            ),
            Tool(
                name="search_conversations",
                description="Find sessions whose description contains all of the given words.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                handler=self._search_conversations,
            ),
            Tool(
                name="create_routine",
                description=(
                    "Save instructions for yourself to run on a timer (e.g. a morning brief) or when "
                    "the user asks. Each run you get the instructions and can start workers across "
                    "any projects; they report back as usual and you decide what to tell the user."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Short name, e.g. 'morning brief'."},
                        "instructions": {
                            "type": "string",
                            "description": "What you should do each run, e.g. which projects to start workers in and what to ask them.",
                        },
                        "cron": {
                            "type": ["string", "null"],
                            "description": (
                                "When to run: a 5-field cron expression in the user's local time, e.g. "
                                "'0 9 * * *' (9am daily) or '30 8 * * 1-5' (8:30am on weekdays). "
                                "null to only run when asked."
                            ),
                        },
                    },
                    "required": ["name", "instructions", "cron"],
                },
                handler=self._create_routine,
            ),
            Tool(
                name="list_routines",
                description="List the saved routines.",
                parameters={"type": "object", "properties": {}},
                handler=self._list_routines,
            ),
            Tool(
                name="delete_routine",
                description="Delete a routine. Sessions it already started are kept.",
                parameters={
                    "type": "object",
                    "properties": {"routine_id": {"type": "integer"}},
                    "required": ["routine_id"],
                },
                handler=self._delete_routine,
            ),
            Tool(
                name="run_routine",
                description="Run a routine now (its timer is unchanged). It starts once this turn ends.",
                parameters={
                    "type": "object",
                    "properties": {"routine_id": {"type": "integer"}},
                    "required": ["routine_id"],
                },
                handler=self._run_routine,
            ),
            self._send_chat_message_tool(),
        ]

    def _send_chat_message_tool(self) -> Tool:
        properties: dict[str, Any] = {"content": {"type": "string"}}
        if choices := self.placement.choices:
            properties["reply_in"] = {
                "type": ["integer", "null"],
                "enum": [*choices, None],
                "description": (
                    "null to post top-level (inline), or the id of a message to post in the thread "
                    "under it (starting the thread if there isn't one). Only these ids are allowed. "
                    "Later updates about this work can go there too."
                ),
            }
        return Tool(
            name="send_chat_message",
            description="Send the user a message in the chat. This is the only way the user sees anything you say.",
            parameters={"type": "object", "properties": properties, "required": list(properties)},
            handler=self.send_chat_message,
        )

    async def _start_conversation(self, prompt: str, description: str, project: str | None = None) -> str:
        if project is None:
            path = self.projects_dir  # general work: every repo is one cd away
        else:
            path = context.project_path(self.projects_dir, project)
            if path is None:
                return f"Error: no project called {project!r}"
        conversation_id = await worker.create_conversation(
            self.engine, prompt, description, path,
            owner_message_id=self.owner_id, on_turn_end=self.on_turn_end,
            routine_run_id=self.routine_run_id,
        )
        self.touched.add(conversation_id)
        return f"Started conversation #{conversation_id}. You'll get an update when it finishes."

    async def _send_to_conversation(self, conversation_id: int, prompt: str) -> str:
        owner_id = resolve_owner(self.engine, conversation_id, self.owner_id)
        try:
            await worker.send_message(
                self.engine, conversation_id, prompt,
                owner_message_id=owner_id, on_turn_end=self.on_turn_end,
            )
        except WorkerBusyError:
            return f"Conversation #{conversation_id} is still running its last turn; wait for its update."
        self.touched.add(conversation_id)
        return f"Sent to conversation #{conversation_id}. You'll get an update when it finishes."

    async def _read_conversation(self, conversation_id: int) -> str:
        with Session(self.engine) as session:
            conversation = session.get(WorkerConversation, conversation_id)
            if conversation is None:
                return f"Error: no conversation #{conversation_id}"
            header = context.describe(conversation)
        return f"{header}\n\n{context.transcript(self.engine, conversation_id, READ_TOKENS)}"

    async def _rename_conversation(self, conversation_id: int, description: str) -> str:
        with Session(self.engine) as session:
            renamed = session.execute(
                update(WorkerConversation)
                .where(WorkerConversation.id == conversation_id)
                .values(description=description)
            ).rowcount
            session.commit()
        return "Renamed." if renamed else f"Error: no conversation #{conversation_id}"

    async def _search_conversations(self, query: str) -> str:
        statement = select(WorkerConversation).order_by(WorkerConversation.last_used.desc()).limit(SEARCH_LIMIT)
        for word in query.split():
            statement = statement.where(WorkerConversation.description.icontains(word, autoescape=True))
        with Session(self.engine) as session:
            conversations = session.scalars(statement).all()
        return "\n".join(context.describe(c) for c in conversations) or "No matches."

    async def _create_routine(self, name: str, instructions: str, cron: str | None) -> str:
        if cron is not None and not croniter.is_valid(cron):
            return f"Error: {cron!r} isn't a valid cron expression"

        with Session(self.engine, expire_on_commit=False) as session:
            routine = Routine(
                name=name, instructions=instructions, cron=cron,
                next_run_at=routines.next_run(cron, routines.utc_now()) if cron else None,
            )
            session.add(routine)
            session.commit()

        if routine.next_run_at is None:
            return f"Created routine #{routine.id}. It only runs when asked."
        return f"Created routine #{routine.id}. First run: {routines.local_time(routine.next_run_at)}."

    async def _list_routines(self) -> str:
        with Session(self.engine) as session:
            rows = session.scalars(
                select(Routine).where(Routine.deleted_at.is_(None)).order_by(Routine.id)
            ).all()
            return "\n\n".join(
                f'#{r.id} "{r.name}", '
                + (f"cron {r.cron!r}, next run {routines.local_time(r.next_run_at)}" if r.cron else "only when asked")
                + f":\n{r.instructions}"
                for r in rows
            ) or "No routines."

    async def _delete_routine(self, routine_id: int) -> str:
        with Session(self.engine) as session:
            routine = routines.get_routine(session, routine_id)
            if routine is None:
                return f"Error: no routine #{routine_id}"
            routine.deleted_at = routines.utc_now()  # soft delete: its runs still refer to it
            session.commit()
        return "Deleted."

    async def _run_routine(self, routine_id: int) -> str:
        with Session(self.engine) as session:
            if routines.get_routine(session, routine_id) is None:
                return f"Error: no routine #{routine_id}"
        # Asked for in the chat, so the run belongs to the current message like any other task.
        self.start_routine(routine_id, self.owner_id)
        return "The routine will run once this turn ends."

    async def send_chat_message(self, content: str, reply_in: int | None = None) -> str:
        if self.placement.choices:
            if reply_in is not None and reply_in not in self.placement.choices:
                return f"Error: reply_in must be null or one of {self.placement.choices}"
            parent_id = reply_in
        else:
            parent_id = self.placement.parent_id  # no choice: reply_in is ignored

        message = chat.send_message(self.engine, "assistant", context.strip_label(content), parent_id)
        self.posted.append(message)
        # Later messages this turn go in the same place.
        self.placement = Placement(parent_id=parent_id)

        if self.owner_id is None:
            # Nobody asked for this work, so this message becomes its owner:
            # later messages and worker prompts this turn go in its thread.
            self.owner_id = message.id
            self.placement = Placement(parent_id=message.id)
        return "Sent."
