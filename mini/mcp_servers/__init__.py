"""MCP servers that give worker sessions extra tools (see `browser.py`).

`worker_mcp_args` returns the `claude` flags that connect a worker to them.
"""

import json
import sys
import tempfile
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Each server runs as `python -m <module>` with this interpreter. Workers run
# in other repos, so PYTHONPATH points `-m` back at this one.
SERVERS = {
    "browser": "mini.mcp_servers.browser",
    "memory": "mini.mcp_servers.memory",
}


def mcp_config() -> dict:
    return {
        "mcpServers": {
            name: {
                "command": sys.executable,
                "args": ["-m", module],
                "env": {"PYTHONPATH": str(REPO_ROOT)},
            }
            for name, module in SERVERS.items()
        }
    }


@cache
def worker_mcp_args() -> list[str]:
    """Flags that load our MCP servers into a `claude` run and pre-approve their tools.

    Claude Code doesn't save MCP config with the session, so pass these on
    every turn, resumes included.
    """
    # Written to a file rather than passed inline to avoid Windows
    # command-line quoting issues with the JSON.
    path = Path(tempfile.gettempdir()) / "mini-mcp.json"
    path.write_text(json.dumps(mcp_config(), indent=2), encoding="utf-8")
    return [
        "--mcp-config", str(path),
        "--allowedTools", *(f"mcp__{name}__*" for name in SERVERS),
    ]
