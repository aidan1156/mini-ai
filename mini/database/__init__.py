from pathlib import Path

from sqlalchemy import Engine, create_engine, event

from mini.database.migrations import run_migrations
from mini.database.models import Base

# <repo root>/database/database.db
DATABASE_PATH = Path(__file__).resolve().parents[2] / "database" / "database.db"


def init_db(database_path: Path = DATABASE_PATH) -> Engine:
    """Run pending migrations and return an engine for the database."""
    run_migrations(database_path)

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")

    # SQLite ignores foreign keys unless this is set on every connection.
    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys = ON")

    return engine


__all__ = ["Base", "DATABASE_PATH", "init_db", "run_migrations"]
