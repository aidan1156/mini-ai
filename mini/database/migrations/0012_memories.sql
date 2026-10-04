-- Where memories come from: a file or a chat message. Many memories can share
-- one source (e.g. every fact from one PDF). A source goes when its file or
-- message does; the memories stay, just without a source.
CREATE TABLE sources (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    attachment_id INTEGER REFERENCES attachments(id) ON DELETE CASCADE,
    message_id    INTEGER REFERENCES chat_messages(id) ON DELETE CASCADE,
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK ((attachment_id IS NOT NULL) + (message_id IS NOT NULL) = 1)  -- exactly one
);

CREATE UNIQUE INDEX idx_sources_attachment ON sources (attachment_id) WHERE attachment_id IS NOT NULL;
CREATE UNIQUE INDEX idx_sources_message ON sources (message_id) WHERE message_id IS NOT NULL;

-- Small atomic facts the orchestrator saves for itself and searches later,
-- by embedding similarity and by full text (see mini/memory.py).
CREATE TABLE memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL,
    embedding       BLOB NOT NULL,  -- float32 vector of `content`, searched with sqlite-vec
    embedding_model TEXT NOT NULL,  -- vectors from different models are not comparable
    source_id       INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Full-text index over `content`, kept in sync by the triggers below.
CREATE VIRTUAL TABLE memories_fts USING fts5(content, content='memories', content_rowid='id');

CREATE TRIGGER memories_after_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts (rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER memories_after_delete AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts (memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER memories_after_update AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts (memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO memories_fts (rowid, content) VALUES (new.id, new.content);
END;
