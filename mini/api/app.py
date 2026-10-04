"""The HTTP API for chatting with mini.

Every request needs `Authorization: Bearer <MINI_API_TOKEN>`.

- POST /messages: send the orchestrator a message (or a voice note, which is
  transcribed first). Returns the saved message straight away; replies
  arrive later on the event stream.
- POST /attachments: upload a file (multipart, field `file`) to send with a
  message: pass the returned id in the message's `attachment_ids`.
- GET /attachments/{id}: download a file sent with a message.
- GET /messages?page=N: the chat, newest first, 20 messages a page.
- GET /events: a Server-Sent Events stream of every new chat message (the
  user's and mini's). Each event's id is the message id: reconnect with a
  `Last-Event-ID` header (or `?after=<id>`) to replay anything missed.
"""

import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.sse import EventSourceResponse, ServerSentEvent
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from mini import attachments, people
from mini.api.public_models import AttachmentOut, ChatMessageOut, MessagePage, SendMessageIn
from mini.config import load_config
from mini.database import init_db
from mini.database.models import ChatMessage
from mini.llm.openai_adapter import OpenAILLM
from mini.orchestrator import chat, context, routines
from mini.orchestrator.orchestrator import Orchestrator
from mini.workers import worker

logger = logging.getLogger(__name__)

PAGE_SIZE = 20

load_dotenv()

_bearer = HTTPBearer(auto_error=False)


def require_token(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]) -> None:
    expected = os.environ["MINI_API_TOKEN"]
    if credentials is None or not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("MINI_API_TOKEN"):
        raise RuntimeError("Set MINI_API_TOKEN in .env")
    projects_dir = load_config().projects_dir
    app.state.engine = init_db()
    # Workers run inside this process, so any still marked running were cut off by a restart.
    worker.mark_interrupted(app.state.engine)
    llm = OpenAILLM()
    app.state.transcriber = llm
    app.state.orchestrator = Orchestrator(app.state.engine, llm, projects_dir, embedder=llm)

    scheduler = asyncio.create_task(
        routines.run_scheduler(app.state.engine, app.state.orchestrator.start_routine)
    )
    yield
    scheduler.cancel()


app = FastAPI(title="mini", lifespan=lifespan, dependencies=[Depends(require_token)])


# async so it runs on the event loop: the orchestrator replies in a background task there.
@app.post("/messages", status_code=status.HTTP_201_CREATED)
async def send_message(body: SendMessageIn, request: Request) -> ChatMessageOut:
    engine = request.app.state.engine
    parent_id = body.parent_id
    if parent_id is not None:
        with Session(engine) as session:
            parent = session.get(ChatMessage, parent_id)
            if parent is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, f"No message {parent_id}")
            parent_id = parent.parent_id or parent.id  # threads are one level deep

    content, voice_note = body.content, None
    if body.voice_note is not None:
        content = await _transcribe(request, body.voice_note.attachment_id)
        voice_note = (body.voice_note.attachment_id, body.voice_note.duration)

    try:
        message = request.app.state.orchestrator.handle_user_message(
            content, parent_id, body.attachment_ids, voice_note,
            sender=(body.sender.provider, body.sender.external_id) if body.sender else None,
        )
    except attachments.AttachmentError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except people.UnknownIdentityError as e:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(e))
    return ChatMessageOut.model_validate(message)


async def _transcribe(request: Request, attachment_id: int) -> str:
    """Transcribe an uploaded, not yet sent recording."""
    recording = attachments.get(request.app.state.engine, attachment_id)
    if recording is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"No attachment {attachment_id}")
    if recording.chat_message_id is not None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Attachment {attachment_id} is already on another message")

    try:
        transcript = await request.app.state.transcriber.transcribe(
            attachments.path(recording).read_bytes(), recording.filename,
            expected_words=["Mini", *context.project_names(load_config().projects_dir)],
        )
    except Exception as e:
        logger.exception("Couldn't transcribe attachment %s", attachment_id)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Couldn't transcribe the voice note: {e}")
    return transcript or "(no speech recognised)"


@app.post("/attachments", status_code=status.HTTP_201_CREATED)
async def upload_attachment(file: UploadFile, request: Request) -> AttachmentOut:
    data = await file.read(attachments.MAX_SIZE + 1)  # one byte over is enough to know it's too big
    try:
        attachment = attachments.save(request.app.state.engine, file.filename or "file", file.content_type, data)
    except attachments.AttachmentError as e:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(e))
    return AttachmentOut.model_validate(attachment)


@app.get("/attachments/{attachment_id}")
def download_attachment(attachment_id: int, request: Request) -> FileResponse:
    attachment = attachments.get(request.app.state.engine, attachment_id)
    if attachment is None or not attachments.path(attachment).is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No attachment {attachment_id}")
    return FileResponse(
        attachments.path(attachment), media_type=attachment.content_type, filename=attachment.filename
    )


@app.get("/messages")
def list_messages(request: Request, page: Annotated[int, Query(ge=1)] = 1) -> MessagePage:
    with Session(request.app.state.engine) as session:
        total = session.scalar(select(func.count(ChatMessage.id)))
        messages = session.scalars(
            select(ChatMessage)
            .order_by(ChatMessage.id.desc())
            .offset((page - 1) * PAGE_SIZE)
            .limit(PAGE_SIZE)
        ).all()
        return MessagePage(
            page=page,
            page_size=PAGE_SIZE,
            total=total,
            messages=[ChatMessageOut.model_validate(m) for m in messages],
        )


@app.get("/events", response_class=EventSourceResponse)
async def events(
    request: Request,
    last_event_id: Annotated[int | None, Header()] = None,
    after: int | None = None,
):
    engine = request.app.state.engine
    # Subscribe before replaying so nothing sent in between is missed.
    queue = chat.subscribe()
    try:
        last_id = after if after is not None else last_event_id
        if last_id is None:
            with Session(engine) as session:
                last_id = session.scalar(select(func.max(ChatMessage.id))) or 0  # only new messages

        with Session(engine) as session:
            missed = session.scalars(
                select(ChatMessage).where(ChatMessage.id > last_id).order_by(ChatMessage.id)
            ).all()
        for message in missed:
            yield _event(message)
            last_id = message.id

        while True:
            message = await queue.get()
            if message.id > last_id:  # skip ones already sent in the replay
                yield _event(message)
                last_id = message.id
    finally:
        chat.unsubscribe(queue)


def _event(message: ChatMessage) -> ServerSentEvent:
    return ServerSentEvent(data=ChatMessageOut.model_validate(message), id=str(message.id))
