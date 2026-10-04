-- The people who send messages (several can share the chat). `name` is the
-- canonical name mini uses; it's set by hand, not synced from any platform.
CREATE TABLE people (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- The accounts a person has on each platform, e.g. ('discord', '123456789') or
-- ('web', 'aidan'). A person can have several, so they're the same person
-- wherever they message from.
CREATE TABLE person_identities (
    provider    TEXT NOT NULL,
    external_id TEXT NOT NULL,
    person_id   INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (provider, external_id)
);

-- Who sent a user message. NULL for the orchestrator's messages, and for user
-- messages from before senders were recorded.
ALTER TABLE chat_messages ADD COLUMN sender_id INTEGER REFERENCES people(id);
