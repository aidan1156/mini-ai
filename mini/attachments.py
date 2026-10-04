"""Files sent with chat messages.

Each file is stored at data/attachments/<id>/<filename>, with its details in
the `attachments` table. Files are uploaded first (unattached) and then
attached when the message they belong to is sent (see chat.send_message).
Workers are given access to this folder, so the orchestrator can hand them
attachments by path.
"""

import mimetypes
import re
from collections.abc import Collection
from pathlib import Path

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from mini.database import DATA_DIR
from mini.database.models import Attachment

ATTACHMENTS_DIR = DATA_DIR / "attachments"
MAX_SIZE = 25 * 1024 * 1024  # bytes per file
FILENAME_LIMIT = 120


class AttachmentError(Exception):
    pass


def path(attachment: Attachment) -> Path:
    return ATTACHMENTS_DIR / str(attachment.id) / attachment.filename


def save(engine: Engine, filename: str, content_type: str | None, data: bytes) -> Attachment:
    """Store a new, not yet attached file."""
    if len(data) > MAX_SIZE:
        raise AttachmentError(f"{filename} is over the {MAX_SIZE // (1024 * 1024)} MB limit")

    filename = _safe_filename(filename)
    content_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"

    with Session(engine, expire_on_commit=False) as session:
        attachment = Attachment(filename=filename, content_type=content_type, size=len(data))
        session.add(attachment)
        session.flush()  # for the id, which names the folder
        file = path(attachment)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
        session.commit()
    return attachment


def save_file(engine: Engine, source: Path) -> Attachment:
    """Store a copy of a file on disk (e.g. one a worker made), so it outlives the original."""
    if source.stat().st_size > MAX_SIZE:
        raise AttachmentError(f"{source.name} is over the {MAX_SIZE // (1024 * 1024)} MB limit")
    return save(engine, source.name, None, source.read_bytes())


def get(engine: Engine, attachment_id: int) -> Attachment | None:
    with Session(engine) as session:
        return session.get(Attachment, attachment_id)


def attach(session: Session, attachment_ids: Collection[int], chat_message_id: int) -> None:
    """Attach uploaded files to a message, in the caller's transaction."""
    if not attachment_ids:
        return
    found = session.scalars(select(Attachment).where(Attachment.id.in_(attachment_ids))).all()
    missing = set(attachment_ids) - {a.id for a in found}
    if missing:
        raise AttachmentError(f"No attachment(s) {sorted(missing)}")
    taken = [a.id for a in found if a.chat_message_id is not None]
    if taken:
        raise AttachmentError(f"Attachment(s) {taken} are already on another message")
    for attachment in found:
        attachment.chat_message_id = chat_message_id


def describe(attachment: Attachment) -> str:
    """One line for the orchestrator: name, type, size, where a worker can read it, and its id."""
    return (
        f"{attachment.filename} ({attachment.content_type}, {_size(attachment.size)}) "
        f"at {path(attachment)}, attachment #{attachment.id}"
    )


def _safe_filename(filename: str) -> str:
    # Keep only the last path part and characters that are safe on every OS.
    name = re.sub(r"[^\w.\-() ]", "_", Path(filename.replace("\\", "/")).name).strip(" .") or "file"
    if len(name) > FILENAME_LIMIT:
        stem, dot, suffix = name.rpartition(".")
        name = (stem[: FILENAME_LIMIT - len(suffix) - 1] + dot + suffix) if dot else name[:FILENAME_LIMIT]
    return name


def _size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"
