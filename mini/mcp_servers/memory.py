"""MCP server that gives workers mini's memory: search, save and update facts.

It's the same code the orchestrator uses (`mini.memory`, `mini.memory_tools`),
running in this process against the same database. Lets a worker read a long
file and save its facts itself instead of reporting them all back.

Runs over stdio. Needs OPENAI_API_KEY (for embeddings and for tidying new facts
against old ones), read from the environment or the repo's `.env`. Workers get
it via `mini.mcp_servers.worker_mcp_args`.
"""

import asyncio
from pathlib import Path

import mcp.types as types
from dotenv import load_dotenv
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from mini.database import init_db
from mini.llm import ToolCall
from mini.llm.openai_adapter import OpenAILLM
from mini.memory import MemoryStore
from mini.memory_tools import memory_tools
from mini.orchestrator.tools import run_tool

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_server(store: MemoryStore) -> Server:
    # Workers can search, save and update, but not delete.
    tools = memory_tools(store, choose_source=True, can_delete=False)
    server = Server("memory")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [types.Tool(name=t.name, description=t.description, inputSchema=t.parameters) for t in tools]

    @server.call_tool(validate_input=False)  # run_tool reports bad arguments back to the model
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        result = await run_tool(tools, ToolCall(id="", name=name, arguments=arguments))
        return [types.TextContent(type="text", text=result)]

    return server


async def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    llm = OpenAILLM()
    server = build_server(MemoryStore(init_db(), embedder=llm, llm=llm))
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
