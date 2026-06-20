import time
import subprocess
from pathlib import Path
from logging import getLogger

logger = getLogger(__name__)


class GeneralAgent:
    def __init__(self, workspace_path: Path) -> None:
        self._workspace_path = workspace_path

    def create_agent(self, prompt: str) -> str:
        cmd = ["agy", "-p", prompt, "--add-dir", str(self._workspace_path)]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            raw_output = result.stdout.strip()

        except subprocess.CalledProcessError as e:
            logger.exception(f"Agent crashed with error: {e.stderr}")
            return "Agent crashed with error:\n" + e.stderr.strip()
        except Exception as e:
            logger.exception(f"Failed to run agent with error: {e}")
            return "Failed to run agent with error:\n" + str(e)
        return raw_output
