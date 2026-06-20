from pydantic import BaseModel

class SendMessageRequest(BaseModel):
    message: str
    conversation_id: int | None = None

class SendMessageResponse(BaseModel):
    response: str
    conversation_id: int
