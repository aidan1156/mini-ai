-- Links a chat message to the Discord message showing it. Only the Discord bot
-- (mini/discord/) uses this table.
CREATE TABLE discord_messages (
    chat_message_id    INTEGER PRIMARY KEY
                       REFERENCES chat_messages(id) ON DELETE CASCADE,
    discord_message_id INTEGER NOT NULL UNIQUE,
    channel_id         INTEGER NOT NULL,  -- the channel (or thread) the message is in
    thread_id          INTEGER,           -- the thread started from this message, once there is one
    created_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_discord_messages_thread ON discord_messages (thread_id);
