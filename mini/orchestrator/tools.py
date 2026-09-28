"""The orchestrator's tools, and running them for one orchestrator turn."""

import logging
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, select, update
from sqlalchemy.orm import Session

from mini.database.models import ChatMessage, WorkerConversation
from mini.llm import Tool, ToolCall
from mini.orchestrator import chat, context
from mini.orchestrator.owners import resolve_owner
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
    send_chat_message posts with `parent_id`, or, if `thread_under` is given,
    the model can choose to start a thread under that message instead. Once
    it has posted, later messages this turn go in the same place.
    """

    def __init__(
        self,
        engine: Engine,
        on_turn_end: TurnEndHandler,
        projects_dir: Path,
        owner_id: int | None,
        parent_id: int | None,
        thread_under: int | None = None,
    ):
        self.engine = engine
        self.on_turn_end = on_turn_end
        self.projects_dir = projects_dir
        self.owner_id = owner_id
        self.parent_id = parent_id
        self.thread_under = thread_under
        self.touched: set[int] = set()  # conversations started or messaged this turn
        self.posted: list[ChatMessage] = []

    def tools(self) -> list[Tool]:
        return [
            Tool(
                name="start_conversation",
                description=(
                    "Start a new Claude Code session to work on a task in one of the projects. It runs "
                    "in the background; you'll get an update when it finishes its turn."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "The task, with all the context the session needs."},
                        "description": {"type": "string", "description": "A short summary of the work, used to find it later."},
                        "project": {"type": "string", "description": "The project (folder name) to work in."},
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
            self._send_chat_message_tool(),
        ]

    def _send_chat_message_tool(self) -> Tool:
        properties: dict[str, Any] = {"content": {"type": "string"}}
        if self.thread_under is not None:
            properties["in_thread"] = {
                "type": "boolean",
                "description": (
                    f"true to start a thread under #{self.thread_under}, for bigger work with several "
                    "updates to come; false to reply inline, for quick answers and small tasks. "
                    "Later updates about this work go wherever this message goes."
                ),
            }
        return Tool(
            name="send_chat_message",
            description="Send the user a message in the chat. This is the only way the user sees anything you say.",
            parameters={"type": "object", "properties": properties, "required": list(properties)},
            handler=self.send_chat_message,
        )

    async def _start_conversation(self, prompt: str, description: str, project: str) -> str:
        path = (self.projects_dir / project).resolve()
        # Only folders directly inside projects_dir (no "..", absolute paths or nesting).
        if path.parent != self.projects_dir or not path.is_dir():
            return f"Error: no project called {project!r}"
        conversation_id = await worker.create_conversation(
            self.engine, prompt, description, path,
            owner_message_id=self.owner_id, on_turn_end=self.on_turn_end,
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

    async def send_chat_message(self, content: str, in_thread: bool = False) -> str:
        parent_id = self.thread_under if in_thread and self.thread_under is not None else self.parent_id
        message = chat.send_message(self.engine, "assistant", context.strip_label(content), parent_id)
        self.posted.append(message)
        self.parent_id, self.thread_under = parent_id, None

        if self.owner_id is None:
            # Nobody asked for this work, so this message becomes its owner:
            # later messages and worker prompts this turn go in its thread.
            self.owner_id = self.parent_id = message.id
        return "Sent."
