import json
import os
from collections.abc import Collection
from typing import Any

from openai import AsyncOpenAI

from mini.llm.base import Message, Tool, ToolCall

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"


class OpenAILLM:
    """OpenAI Responses API. Reads OPENAI_API_KEY (and OPENAI_MODEL) from the environment.

    Nothing is stored on OpenAI's side (`store=False`), so each call sends the
    whole conversation. The model's reasoning comes back encrypted and is kept
    on the assistant Message (`provider_data`) so it can be sent back on the
    next call of a tool loop.

    Also transcribes audio (a Transcriber), with OPENAI_TRANSCRIBE_MODEL.
    """

    def __init__(self, model: str | None = None, client: AsyncOpenAI | None = None):
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.transcribe_model = os.environ.get("OPENAI_TRANSCRIBE_MODEL", DEFAULT_TRANSCRIBE_MODEL)
        self.client = client or AsyncOpenAI()

    async def transcribe(self, audio: bytes, filename: str, expected_words: Collection[str] = ()) -> str:
        kwargs: dict[str, Any] = {}
        if expected_words:
            # The prompt steers spelling: names in it come back spelled the same way.
            kwargs["prompt"] = f"Names that may come up: {', '.join(expected_words)}."
        transcription = await self.client.audio.transcriptions.create(
            model=self.transcribe_model, file=(filename, audio), **kwargs
        )
        return transcription.text.strip()

    async def complete(self, system: str, messages: list[Message], tools: list[Tool]) -> Message:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]

        response = await self.client.responses.create(
            model=self.model,
            instructions=system,
            input=[item for message in messages for item in _to_input_items(message)],
            store=False,
            include=["reasoning.encrypted_content"],
            **kwargs,
        )
        return _from_output(response.output)


def _to_openai_tool(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


def _to_input_items(message: Message) -> list[dict[str, Any]]:
    if message.role == "tool":
        return [{"type": "function_call_output", "call_id": message.tool_call_id, "output": message.content or ""}]

    if message.role == "assistant" and message.provider_data is not None:
        return message.provider_data  # the original output items, reasoning included

    items: list[dict[str, Any]] = []
    if message.content:
        items.append({"role": message.role, "content": message.content})
    for call in message.tool_calls:
        items.append({
            "type": "function_call",
            "call_id": call.id,
            "name": call.name,
            "arguments": json.dumps(call.arguments),
        })
    return items


def _from_output(output: list[Any]) -> Message:
    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    for item in output:
        if item.type == "message":
            texts += [part.text for part in item.content if part.type == "output_text"]
        elif item.type == "function_call":
            try:
                arguments = json.loads(item.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}  # the tool will report the missing arguments back to the model
            tool_calls.append(ToolCall(id=item.call_id, name=item.name, arguments=arguments))

    return Message(
        role="assistant",
        content="\n".join(texts) or None,
        tool_calls=tool_calls,
        provider_data=[item.model_dump(exclude_none=True) for item in output],
    )
