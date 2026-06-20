import os
import base64
from sqlalchemy import Engine
from google import genai
from google.genai import types

from mini.api_adapters.llm_adapter import LLMConversationAdapter

def _encode_bytes(obj):
    if isinstance(obj, dict):
        return {k: _encode_bytes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_encode_bytes(x) for x in obj]
    elif isinstance(obj, bytes):
        return base64.b64encode(obj).decode('utf-8')
    return obj

class GeminiConversationAdapter(LLMConversationAdapter):
    def __init__(
        self,
        engine: Engine,
        conversation_id: int | None = None,
        model: str = "gemini-3.5-flash"
    ):
        super().__init__(engine, conversation_id)
        self.model = model
        api_key = os.environ.get("GEMINI_API") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            self._client = genai.Client(api_key=api_key)
        else:
            self._client = genai.Client()

    def _send_message_impl(
        self,
        message: str, 
        history: list[dict],
        system_prompt: str | None = None,
        tools: list | None = None
    ) -> tuple[str, list[dict]]:
        history_to_load = []
        for msg in history:
            if "parts" in msg:
                try:
                    history_to_load.append(types.Content.model_validate(msg))
                except Exception:
                    pass
            elif "text" in msg:
                part = types.Part.from_text(text=msg["text"])
                role = msg.get("role", "user")
                if role == "user":
                    history_to_load.append(types.UserContent(parts=[part]))
                else:
                    history_to_load.append(types.ModelContent(parts=[part]))

        config = None
        if system_prompt is not None or tools is not None:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=tools
            )

        chat = self._client.chats.create(
            model=self.model,
            history=history_to_load,
            config=config
        )
        
        response = chat.send_message(message)
        
        # Serialize the updated history including the model's function calls and results
        raw_history = [content.model_dump() for content in chat.get_history()]
        updated_history = _encode_bytes(raw_history)
        
        return response.text, updated_history
