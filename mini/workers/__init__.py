"""SQL migrations for the SQLite database.

Add a new file to this folder for each schema change, named with an
increasing number prefix so they sort in order:

    0001_create_skills.sql
    0002_add_conversations.sql

Each file runs once, inside a transaction, and is recorded in the
`schema_migrations` table. Never edit a migration that has already run --
add a new one instead.
"""

import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).resolve().parent


def run_migrations(database_path: Path) -> list[str]:
    """Apply any pending migrations. Returns the names of those applied."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    applied: list[str] = []

    conn = sqlite3.connect(database_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

        done = {row[0] for row in conn.execute("SELECT name FROM schema_migrations")}

        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in done:
                continue

            sql = path.read_text(encoding="utf-8")
            try:
                # executescript runs outside Python's implicit transaction
                # handling, so wrap the whole file explicitly to keep it atomic.
                conn.executescript(
                    f"BEGIN;\n{sql}\n;"
                    f"INSERT INTO schema_migrations (name) VALUES ('{path.name}');\n"
                    "COMMIT;"
                )
            except Exception as e:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise RuntimeError(f"Migration {path.name} failed: {e}") from e

            applied.append(path.name)
    finally:
        conn.close()

    return applied
