"""The orchestrator agent: the one agent the user chats with.

It runs a turn in two cases:

- `handle_user_message`: the user said something. For a top-level message it
  chooses whether to reply inline or in a thread under it.
- `on_worker_turn_end`: a worker session finished a turn. The orchestrator is
  shown the result and decides whether to tell the user. Its message goes
  wherever the conversation about that work already is (see
  mini.orchestrator.owners).

Either way it talks to the user with its send_chat_message tool.

Turns run one at a time. Worker updates that arrive while a turn is running
are batched, one orchestrator turn per owner message.
"""

import asyncio
import datetime
import logging
from collections import defaultdict
from collections.abc import Collection

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from mini.database.models import ChatMessage, WorkerConversation
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
- The user's messages are prefixed with [#id] or [#id in thread #root]. Don't
  write these prefixes yourself.
- Be brief.

Recently used worker conversations:
{conversations}
"""


class Orchestrator:
    def __init__(self, engine: Engine, llm: LLM):
        self.engine = engine
        self.llm = llm
        self._lock = asyncio.Lock()
        self._pending: set[int] = set()  # conversations with unhandled updates
        self._drain_task: asyncio.Task | None = None

    async def handle_user_message(self, content: str, parent_id: int | None = None) -> list[ChatMessage]:
        """Save the user's message, run the orchestrator, and return the messages it sent."""
        user_message = chat.send_message(self.engine, "user", content, parent_id)

        async with self._lock:
            messages = context.chat_history(self.engine, user_message, HISTORY_TOKENS, exclude={user_message.id})
            messages.append(Message(role="user", content=f"{context.label(user_message)} {content}"))

            # Reply next to the user's message, or (if it's top-level) optionally in a thread under it.
            runner = ToolRunner(
                self.engine, self.on_worker_turn_end, owner_id=user_message.id,
                parent_id=user_message.parent_id,
                thread_under=user_message.id if user_message.parent_id is None else None,
            )
            final_text = await self._run(messages, runner)

            # If it answered in plain text instead of using the tool, send that.
            if final_text and not runner.posted:
                await runner.send_chat_message(final_text)

        return runner.posted

    async def on_worker_turn_end(self, conversation_id: int) -> None:
        """Pass to workers as `on_turn_end`. Queues the update and returns straight away."""
        self._pending.add(conversation_id)
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        while self._pending:
            async with self._lock:
                batch, self._pending = self._pending, set()

                by_owner: dict[int | None, list[int]] = defaultdict(list)
                for conversation_id in sorted(batch):
                    by_owner[resolve_owner(self.engine, conversation_id)].append(conversation_id)

                for owner_id, conversation_ids in by_owner.items():
                    try:
                        await self._handle_update(owner_id, conversation_ids)
                    except Exception:
                        # Nobody awaits this task, so log and carry on with the rest.
                        logger.exception("Orchestrator failed on update for conversations %s", conversation_ids)

    async def _handle_update(self, owner_id: int | None, conversation_ids: list[int]) -> None:
        with Session(self.engine, expire_on_commit=False) as session:
            owner = session.get(ChatMessage, owner_id) if owner_id is not None else None

        messages = context.chat_history(self.engine, owner, HISTORY_TOKENS)
        messages.append(Message(role="user", content=self._update_text(owner, conversation_ids)))

        runner = ToolRunner(
            self.engine, self.on_worker_turn_end, owner_id=owner_id,
            parent_id=update_parent(self.engine, owner),
        )
        await self._run(messages, runner)  # plain final text is ignored: saying nothing is allowed

        if owner is None and runner.posted:
            adopt(self.engine, runner.posted[0].id, {*conversation_ids, *runner.touched})

    def _update_text(self, owner: ChatMessage | None, conversation_ids: Collection[int]) -> str:
        with Session(self.engine) as session:
            conversations = [session.get(WorkerConversation, i) for i in conversation_ids]

        parts = ["<worker_update>", "This is an automatic update, not a message from the user.", ""]
        for conversation in conversations:
            parts += [
                f"Conversation {context.describe(conversation)} finished a turn.",
                "Transcript (prompts sent and results):",
                context.transcript(self.engine, conversation.id, UPDATE_TRANSCRIPT_TOKENS, types=("prompt", "result")),
                "",
            ]

        if owner is None:
            parts.append(
                "Nobody asked for this work in the chat. If you send a message it will be posted "
                "top-level, and later updates about this work will go in its thread."
            )
        else:
            parts.append(f"This work was asked for in chat message {context.label(owner)}.")
            others = [c for c in context.owned_conversations(self.engine, owner.id) if c.id not in conversation_ids]
            if others:
                parts.append("Other conversations working for it:")
                parts += [f"- {context.describe(c)}" for c in others]

            parent_id = update_parent(self.engine, owner)
            where = f"in thread #{parent_id}" if parent_id is not None else "top-level, like your earlier reply"
            parts.append(f"Messages you send with send_chat_message will go {where}.")

        parts.append(
            "If nothing needs saying yet (e.g. other conversations are still working on the "
            "same request), end your turn without sending anything. You can also react by "
            "sending workers follow-ups."
        )
        parts.append("</worker_update>")
        return "\n".join(parts)

    async def _run(self, messages: list[Message], runner: ToolRunner) -> str | None:
        """The tool-use loop. Returns the model's final text."""
        system = SYSTEM_PROMPT.format(now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), conversations=context.recent_conversations(self.engine, RECENT_CONVERSATIONS))
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
