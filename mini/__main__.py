"""Run everything: python -m mini

Starts the API server and (if "discord" is true in config.json) the Discord
bot as separate processes, with their output prefixed [api] / [discord]. If one
exits it's restarted after a few seconds; the other keeps running. Ctrl+C stops
both. Each can still be run on its own with python -m mini.api / mini.discord.

MCP servers aren't started here: Claude Code launches them for each worker.
"""

import asyncio
import os
import sys
from pathlib import Path

from mini.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
RESTART_SECONDS = 5
# On Ctrl+C the children get it too; give them this long to shut down cleanly
# (the API server stops its running workers) before they're killed.
SHUTDOWN_SECONDS = 10


async def supervise(name: str, module: str) -> None:
    """Run `python -m module`, printing its output, and restart it whenever it exits."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    while True:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", module,
            cwd=REPO_ROOT, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            async for line in process.stdout:
                print(f"[{name}] {line.decode('utf-8', errors='replace').rstrip()}", flush=True)
            code = await process.wait()
        except asyncio.CancelledError:
            await stop(process)
            raise

        print(f"[{name}] exited with code {code}; restarting in {RESTART_SECONDS}s", flush=True)
        await asyncio.sleep(RESTART_SECONDS)


async def stop(process: asyncio.subprocess.Process) -> None:
    try:
        await asyncio.wait_for(process.wait(), SHUTDOWN_SECONDS)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


async def main() -> None:
    services = {"api": "mini.api"}
    if load_config().discord:
        services["discord"] = "mini.discord"

    await asyncio.gather(*(supervise(name, module) for name, module in services.items()))


if __name__ == "__main__":
    # Don't crash on characters the console can't show (e.g. emoji in logs).
    sys.stdout.reconfigure(errors="replace")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
