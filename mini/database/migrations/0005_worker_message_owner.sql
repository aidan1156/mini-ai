-- The chat message the worker is acting on behalf of (usually the user's request
-- that started the turn). When the worker says something, the orchestrator can
-- reply in that message's thread. Kept as NULL if the chat message is deleted.
ALTER TABLE worker_messages
    ADD COLUMN owner_message_id INTEGER REFERENCES chat_messages(id) ON DELETE SET NULL;

CREATE INDEX idx_worker_messages_owner ON worker_messages (owner_message_id);
