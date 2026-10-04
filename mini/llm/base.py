"""A provider-agnostic interface for chat LLMs with tool calling.

The orchestrator only talks to `LLM`; each provider gets an adapter that
converts these types to and from its own API format.
"""

from collections.abc import Awaitable, Callable, Collection
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


class Embedder(Protocol):
    # Names the model, since vectors from different models can't be compared.
    embedding_model: str

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text, in order."""
        ...


class Transcriber(Protocol):
    async def transcribe(self, audio: bytes, filename: str, expected_words: Collection[str] = ()) -> str:
        """Return the text spoken in `audio` (the filename's extension gives its format).

        `expected_words` are names likely to come up, to help spell them right.
        """
        ...
