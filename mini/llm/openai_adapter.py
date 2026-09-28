import json
import os
from typing import Any

from openai import AsyncOpenAI

from mini.llm.base import Message, Tool, ToolCall

DEFAULT_MODEL = "gpt-5-mini"


class OpenAILLM:
    """OpenAI Chat Completions. Reads OPENAI_API_KEY (and OPENAI_MODEL) from the environment."""

    def __init__(self, model: str | None = None, client: AsyncOpenAI | None = None):
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.client = client or AsyncOpenAI()

    async def complete(self, system: str, messages: list[Message], tools: list[Tool]) -> Message:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *map(_to_openai_message, messages)],
            **kwargs,
        )
        return _from_openai_message(response.choices[0].message)


def _to_openai_tool(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _to_openai_message(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content or ""}

    result: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return result


def _from_openai_message(message: Any) -> Message:
    tool_calls = []
    for call in message.tool_calls or []:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}  # the tool will report the missing arguments back to the model
        tool_calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
    return Message(role="assistant", content=message.content, tool_calls=tool_calls)
