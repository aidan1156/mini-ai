-- Instructions for the orchestrator that run on a timer (e.g. a morning brief)
-- or when asked. A run is a hidden orchestrator turn: nothing is posted to the
-- chat unless the orchestrator decides to.
CREATE TABLE routines (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    instructions TEXT NOT NULL,  -- for the orchestrator, not a worker
    cron         TEXT,           -- 5-field cron in the server's local time; NULL = only run when asked
    enabled      INTEGER NOT NULL DEFAULT 1,
    next_run_at  TIMESTAMP,      -- UTC, like CURRENT_TIMESTAMP; NULL without a cron
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_routines_due ON routines (enabled, next_run_at);

-- One row per time a routine ran, so the workers it started can be grouped.
CREATE TABLE routine_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    routine_id   INTEGER REFERENCES routines(id) ON DELETE SET NULL,
    routine_name TEXT NOT NULL,  -- kept in case the routine is deleted
    started_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- The routine run that started this conversation, if any.
ALTER TABLE worker_conversations
    ADD COLUMN routine_run_id INTEGER REFERENCES routine_runs(id) ON DELETE SET NULL;

CREATE INDEX idx_worker_conversations_routine_run ON worker_conversations (routine_run_id);
