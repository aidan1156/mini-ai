from pathlib import Path

import sqlite_vec
from sqlalchemy import Engine, create_engine, event

from mini.database.migrations import run_migrations
from mini.database.models import Base

# <repo root>/data holds everything mini stores (git-ignored).
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATABASE_PATH = DATA_DIR / "database.db"


def init_db(database_path: Path = DATABASE_PATH) -> Engine:
    """Run pending migrations and return an engine for the database."""
    run_migrations(database_path)

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")

    @event.listens_for(engine, "connect")
    def _setup_connection(dbapi_conn, _):
        # SQLite ignores foreign keys unless this is set on every connection.
        dbapi_conn.execute("PRAGMA foreign_keys = ON")
        # sqlite-vec's vector functions, for searching memories.
        dbapi_conn.enable_load_extension(True)
        sqlite_vec.load(dbapi_conn)
        dbapi_conn.enable_load_extension(False)

    return engine


__all__ = ["Base", "DATA_DIR", "DATABASE_PATH", "init_db", "run_migrations"]
