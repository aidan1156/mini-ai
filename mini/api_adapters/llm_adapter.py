from abc import ABC, abstractmethod
import json

from sqlalchemy import Engine
from sqlalchemy.orm import Session
from mini.database import Conversation

class LLMConversationAdapter(ABC):
    def __init__(self, engine: Engine, conversation_id: int | None = None):
        self._engine = engine
        if conversation_id is None:
            self.conversation_id = self._create_conversation()
        else:
            self.conversation_id = conversation_id

    def _create_conversation(self) -> int:
        with Session(self._engine) as session:
            conversation = Conversation(conversation="[]")
            session.add(conversation)
            session.commit()
            return conversation.id

    def _load_history(self) -> list[dict]:
        with Session(self._engine) as session:
            conversation = session.query(Conversation).filter(Conversation.id == self.conversation_id).first()
            if conversation is None:
                return []
            try:
                return json.loads(conversation.conversation)
            except Exception:
                return []

    def _save_history(self, history: list[dict]) -> None:
        with Session(self._engine) as session:
            conversation = session.query(Conversation).filter(Conversation.id == self.conversation_id).first()
            if conversation is not None:
                conversation.conversation = json.dumps(history)
                session.commit()

    @abstractmethod
    def _send_message_impl(
        self,
        message: str, 
        history: list[dict],
        system_prompt: str | None = None,
        tools: list | None = None
    ) -> tuple[str, list[dict]]:
        """
        Send message to LLM using history.
        Should return a tuple of (model_response_text, updated_history_list).
        """
        pass

    def send_message(
        self,
        message: str, 
        system_prompt: str | None = None,
        tools: list | None = None
    ) -> str:
        # 1. Reload from db before each message
        history = self._load_history()
        
        # 2. Get response and updated history from the actual LLM (implemented by subclass)
        response, updated_history = self._send_message_impl(
            message,
            history,
            system_prompt=system_prompt,
            tools=tools
        )
        
        # 3. Sync to db after every message
        self._save_history(updated_history)
        
        return response

    @staticmethod
    def create(
        engine: Engine,
        conversation_id: int | None = None,
        model: str = "gemini-3.5-flash"
    ) -> "LLMConversationAdapter":
        from mini.api_adapters.gemini_adapter import GeminiConversationAdapter
        return GeminiConversationAdapter(engine, conversation_id=conversation_id, model=model)
