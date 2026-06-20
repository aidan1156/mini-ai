
from pathlib import Path

from pydantic import BaseModel


class ServerConfig(BaseModel):
    host: str
    port: int



class Config(BaseModel):
    server: ServerConfig
    database_path: Path
    workspace_path: Path

def load_config(path: str = 'config.json') -> Config:
    with open(path, "r") as f:
        config_data = f.read()
    return Config.model_validate_json(config_data)