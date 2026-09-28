"""Settings from config.json at the repo root (secrets live in .env instead)."""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


@dataclass(frozen=True)
class Config:
    # Workers only run inside the repos in this folder, one repo per worker.
    projects_dir: Path


@cache
def load_config(path: Path = CONFIG_PATH) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))

    projects_dir = Path(raw["projects_dir"]).resolve()
    if not projects_dir.is_dir():
        raise RuntimeError(f"projects_dir in {path} isn't a directory: {projects_dir}")

    return Config(projects_dir=projects_dir)
