-- The user's chat with the orchestrator agent.
CREATE TABLE chat_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
