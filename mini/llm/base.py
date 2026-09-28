"""A provider-agnostic interface for chat LLMs with tool calling.

The orchestrator only talks to `LLM`; each provider gets an adapter that
converts these types to and from its own API format.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the arguments object
    # Called with the model's arguments as keyword args; returns the result for the model.
    # Adapters only send the fields above to the provider.
    handler: Callable[..., Awaitable[str]]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant only
    tool_call_id: str | None = None  # tool only: the call this is the result of
    # Adapter-specific data to send back on later calls (e.g. the model's reasoning).
    # Only the adapter that created the message reads it.
    provider_data: Any = None


class LLM(Protocol):
    async def complete(self, system: str, messages: list[Message], tools: list[Tool]) -> Message:
        """Return the model's next assistant message."""
        ...
