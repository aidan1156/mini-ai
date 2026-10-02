-- Marks a chat message as a voice note: its content is the transcript, and the
-- recording is one of its attachments.
CREATE TABLE voice_notes (
    chat_message_id INTEGER PRIMARY KEY REFERENCES chat_messages(id) ON DELETE CASCADE,
    attachment_id   INTEGER NOT NULL UNIQUE REFERENCES attachments(id),  -- the recording
    duration        REAL,  -- seconds, if the client knew it
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
