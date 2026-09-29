-- Files sent with chat messages (by the user or the orchestrator). The file
-- itself is stored at data/attachments/<id>/<filename>, not in the database.
CREATE TABLE attachments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    -- NULL between uploading the file and sending the message it's attached to.
    chat_message_id INTEGER REFERENCES chat_messages(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    content_type    TEXT NOT NULL,
    size            INTEGER NOT NULL,  -- bytes
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_attachments_chat_message ON attachments (chat_message_id);
