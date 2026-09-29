import logging
import shutil
from pathlib import Path

from sqlalchemy import Engine, create_engine, event

from mini.database.migrations import run_migrations
from mini.database.models import Base

logger = logging.getLogger(__name__)

# <repo root>/data holds everything mini stores (git-ignored).
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATABASE_PATH = DATA_DIR / "database.db"
# Where the database lived before data/; moved over on startup (see init_db).
LEGACY_DATABASE_PATH = Path(__file__).resolve().parents[2] / "database" / "database.db"


def init_db(database_path: Path = DATABASE_PATH) -> Engine:
    """Run pending migrations and return an engine for the database."""
    if database_path == DATABASE_PATH:
        _move_legacy_database()
    run_migrations(database_path)

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")

    # SQLite ignores foreign keys unless this is set on every connection.
    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys = ON")

    return engine


def _move_legacy_database() -> None:
    """Move database/database.db to data/ if it's still in the old place."""
    if not LEGACY_DATABASE_PATH.exists():
        return
    if DATABASE_PATH.exists():
        logger.warning(
            "Both %s and %s exist; using the second. Merge or delete the old one.",
            LEGACY_DATABASE_PATH, DATABASE_PATH,
        )
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(LEGACY_DATABASE_PATH, DATABASE_PATH)
    try:
        LEGACY_DATABASE_PATH.parent.rmdir()  # only if nothing else is in there
    except OSError:
        pass
    logger.info("Moved %s to %s", LEGACY_DATABASE_PATH, DATABASE_PATH)


__all__ = ["Base", "DATA_DIR", "DATABASE_PATH", "init_db", "run_migrations"]
