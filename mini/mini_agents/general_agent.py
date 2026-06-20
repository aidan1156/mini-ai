import time
import subprocess
from pathlib import Path


class GeneralAgent:
    def __init__(self, workspace_path: Path) -> None:
        self._workspace_path = workspace_path

    def create_agent(self, prompt: str) -> str:
        print(f"Creating general agent with prompt:\n{prompt}\n")

        # cmd = ["agy", "-p", prompt, '--add-dir', str(self._workspace_path)]
        cmd = ["echo", "Done! I have created that for you"]

        print(f"Spawning agent in workspace: {self._workspace_path} ...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                shell=True,
            )
            
            raw_output = result.stdout.strip()
            
        except subprocess.CalledProcessError as e:
            return "Agent crashed with error:\n" + e.stderr.strip()
        except Exception as e:
            return "Failed to run agent with error:\n" + str(e)
        
        return raw_output
