"""The projects folder workers run in."""

from pathlib import Path


def projects(projects_dir: Path) -> str:
    """The project folders workers can run in, one per line."""
    names = sorted(
        (p.name for p in projects_dir.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=str.lower,
    )
    return "\n".join(f"- {name}" for name in names) or "(none yet)"


def project_path(projects_dir: Path, project: str) -> Path | None:
    """The folder for `project`, or None unless it's a folder directly inside projects_dir."""
    path = (projects_dir / project).resolve()
    # No "..", absolute paths or nesting.
    if path.parent != projects_dir or not path.is_dir():
        return None
    return path
