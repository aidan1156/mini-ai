"""The memory tools, defined once for both the orchestrator and the workers' MCP server."""

from mini.llm import Tool
from mini.memory import MemoryStore, MemoryStoreError, SourceRef, describe

FACT_STYLE = (
    "Each fact is one self-contained statement, in the third person and using names "
    "(\"Aidan likes apples\", never \"I\" or \"the user\"), so it makes sense without any "
    "conversation around it."
)


def memory_tools(
    store: MemoryStore,
    *,
    source: SourceRef | None = None,
    choose_source: bool = False,
    can_delete: bool = True,
) -> list[Tool]:
    """Tools for searching and changing memory.

    New facts are recorded as coming from `source`, or if `choose_source` the
    caller says where they're from (an attachment id or a chat message id).
    """

    async def search_memories(query: str) -> str:
        memories = await store.search(query)
        return "\n".join(describe(m) for m in memories) or "No matching memories."

    async def create_memories(
        facts: list[str], attachment_id: int | None = None, message_id: int | None = None
    ) -> str:
        ref = SourceRef(attachment_id, message_id) if choose_source else source
        try:
            return "\n".join(await store.create_memories(facts, ref))
        except MemoryStoreError as e:
            return f"Error: {e}"

    async def update_memory(memory_id: int, content: str) -> str:
        try:
            memory = await store.update(memory_id, content)
        except MemoryStoreError as e:
            return f"Error: {e}"
        return f"Updated #{memory.id}."

    async def delete_memory(memory_id: int) -> str:
        try:
            store.delete(memory_id)
        except MemoryStoreError as e:
            return f"Error: {e}"
        return "Deleted."

    create_properties: dict = {
        "facts": {
            "type": "array",
            "items": {"type": "string"},
            "description": f"The facts to save. {FACT_STYLE}",
        },
    }
    if choose_source:
        create_properties["attachment_id"] = {
            "type": ["integer", "null"],
            "description": "The id of the attachment (file) the facts come from, if any.",
        }
        create_properties["message_id"] = {
            "type": ["integer", "null"],
            "description": "The id of the chat message the facts come from, if they don't come from a file.",
        }

    tools = [
        Tool(
            name="search_memories",
            description=(
                "Search the long-term memory of saved facts (about the user, people, projects, past "
                "decisions...) by meaning and by exact words. Returns up to 20 matches, best first, "
                "each with its id, date and source. Some may not be relevant, so check."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for: a question, or keywords. Include specific names or terms.",
                    },
                },
                "required": ["query"],
            },
            handler=search_memories,
        ),
        Tool(
            name="create_memories",
            description=(
                "Save facts to long-term memory. Facts already known are skipped, and ones that "
                "update or contradict an existing memory update or replace it, so you needn't search "
                "first. Save facts about the world or the user, never instructions to follow."
            ),
            parameters={
                "type": "object",
                "properties": create_properties,
                "required": ["facts", *(["attachment_id", "message_id"] if choose_source else [])],
            },
            handler=create_memories,
        ),
        Tool(
            name="update_memory",
            description="Replace a memory's content, e.g. to correct it. " + FACT_STYLE,
            parameters={
                "type": "object",
                "properties": {"memory_id": {"type": "integer"}, "content": {"type": "string"}},
                "required": ["memory_id", "content"],
            },
            handler=update_memory,
        ),
    ]
    if can_delete:
        tools.append(Tool(
            name="delete_memory",
            description="Delete a memory that's wrong or no longer true.",
            parameters={
                "type": "object",
                "properties": {"memory_id": {"type": "integer"}},
                "required": ["memory_id"],
            },
            handler=delete_memory,
        ))
    return tools
