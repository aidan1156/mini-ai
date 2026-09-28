-- NULL for a top-level message; otherwise the id of the message that started the thread.
ALTER TABLE chat_messages
    ADD COLUMN parent_id INTEGER REFERENCES chat_messages(id) ON DELETE CASCADE;

CREATE INDEX idx_chat_messages_parent ON chat_messages (parent_id, id);
