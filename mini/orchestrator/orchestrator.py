"""The orchestrator agent: the one agent the user chats with.

It runs a turn in two cases:

- `handle_user_message`: the user said something. Its final text is posted as
  the reply, next to the user's message (in the same thread, if any).
- `on_worker_turn_end`: a worker session finished a turn. The orchestrator is
  shown the result and decides whether to tell the user, via its
  send_chat_message tool (see mini.orchestrator.owners for where it goes).

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
from mini.orchestrator.owners import adopt, can_choose_thread, resolve_owner
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
- Chat messages are prefixed with [#id] or [#id in thread #root] so you can tell
  them apart. Don't write these prefixes yourself.
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

    async def handle_user_message(self, content: str, parent_id: int | None = None) -> ChatMessage | None:
        """Save the user's message, run the orchestrator, and return its reply (if any)."""
        user_message = chat.send_message(self.engine, "user", content, parent_id)

        async with self._lock:
            messages = context.chat_history(self.engine, user_message, HISTORY_TOKENS, exclude={user_message.id})
            messages.append(Message(role="user", content=f"{context.label(user_message)} {content}"))

            runner = ToolRunner(self.engine, self.on_worker_turn_end, owner_id=user_message.id)
            reply = await self._run(messages, runner)

        if not reply:
            return None
        return chat.send_message(self.engine, "assistant", reply, user_message.parent_id)

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
            can_post=True, update_owner=owner,
        )
        await self._run(messages, runner)  # it talks to the user via send_chat_message

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

            if can_choose_thread(owner):
                parts.append(
                    f"To tell the user, call send_chat_message with in_thread=true to reply in the "
                    f"thread under #{owner.id} (for anything substantial), or in_thread=false to "
                    f"post top-level (for a quick note)."
                )
            else:
                parts.append(f"Messages you send will go in thread #{owner.parent_id}.")

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
