"""Apply pending migrations to the database.

Run from the repo root:

    python -m mini.database.migrate
"""

from mini.database import DATABASE_PATH
from mini.database.migrations import run_migrations


def main() -> None:
    applied = run_migrations(DATABASE_PATH)

    if applied:
        for name in applied:
            print(f"Applied {name}")
    else:
        print("Database is up to date.")


if __name__ == "__main__":
    main()
