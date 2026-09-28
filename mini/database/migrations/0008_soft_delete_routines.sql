-- Routines are soft-deleted now (deleted_at), so a run can always look up its
-- routine's name instead of keeping a copy.
ALTER TABLE routines ADD COLUMN deleted_at TIMESTAMP;

-- Runs whose routine was hard-deleted before this can't point at one any more.
UPDATE worker_conversations
SET routine_run_id = NULL
WHERE routine_run_id IN (SELECT id FROM routine_runs WHERE routine_id IS NULL);

-- Rebuild routine_runs without routine_name and with routine_id required
-- (SQLite can't add NOT NULL to an existing column).
CREATE TABLE routine_runs_new (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    routine_id INTEGER NOT NULL REFERENCES routines(id),
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO routine_runs_new (id, routine_id, started_at)
SELECT id, routine_id, started_at FROM routine_runs WHERE routine_id IS NOT NULL;

DROP TABLE routine_runs;
ALTER TABLE routine_runs_new RENAME TO routine_runs;
