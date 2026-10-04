import asyncio
import hashlib
import re
from pathlib import Path

import pytest
from mcp import types
from sqlalchemy.orm import Session

from mini import attachments
from mini.database import init_db
from mini.database.models import ChatMessage, Memory, Source
from mini.llm import Message, ToolCall
from mini.mcp_servers.memory import build_server
from mini.memory import MemoryStore, MemoryStoreError, SourceRef, describe
from mini.orchestrator import chat
from mini.orchestrator.owners import Placement
from mini.orchestrator.tools import ToolRunner, run_tool

DIMENSIONS = 64


class FakeEmbedder:
    """Bag of words: texts sharing words are close."""

    embedding_model = "fake"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * DIMENSIONS
            for word in re.findall(r"\w+", text.lower()):
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMENSIONS] += 1
            vectors.append(vector)
        return vectors


class FakeLLM:
    """Answers each consolidation with the next queued decision (none queued: no tool call)."""

    def __init__(self):
        self.decisions: list[dict] = []
        self.calls: list[str] = []

    async def complete(self, system, messages, tools) -> Message:
        self.calls.append(messages[-1].content)
        if not self.decisions:
            return Message(role="assistant", content="ok")
        call = ToolCall(id="1", name="decide", arguments=self.decisions.pop(0))
        return Message(role="assistant", tool_calls=[call])


def decision(action="add", memory_id=None, content=None, delete_ids=()):
    return {"action": action, "memory_id": memory_id, "content": content, "delete_ids": list(delete_ids)}


@pytest.fixture
def engine(tmp_path: Path):
    return init_db(tmp_path / "test.db")


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def store(engine, llm):
    return MemoryStore(engine, FakeEmbedder(), llm)


def run(coroutine):
    return asyncio.run(coroutine)


def contents(engine) -> list[str]:
    with Session(engine) as session:
        return [m.content for m in session.query(Memory).order_by(Memory.id)]


def test_saves_and_finds_by_meaning(store):
    run(store.create_memories(["Aidan likes apples", "The deploy server is in London"]))

    found = run(store.search("what fruit does Aidan like apples"))

    assert found[0].content == "Aidan likes apples"


def test_finds_odd_keywords_by_text(store):
    run(store.create_memories(["The zxqv flag disables the cache", "Aidan likes apples"]))

    assert "The zxqv flag disables the cache" in [m.content for m in run(store.search("zxqv"))]


def test_search_returns_at_most_20(store):
    run(store.create_memories([f"fact number {i} about apples" for i in range(30)]))

    assert len(run(store.search("apples"))) == 20


def test_search_ignores_fts_syntax_in_the_query(store):
    run(store.create_memories(["Aidan likes apples"]))

    assert run(store.search('apples" OR (NEAR')) != []


def test_update_and_delete_change_what_search_finds(engine, store):
    run(store.create_memories(["Aidan likes pears"]))

    run(store.update(1, "Aidan likes quinces"))
    assert run(store.search("quinces"))[0].content == "Aidan likes quinces"
    assert run(store.search("pears"))[0].content == "Aidan likes quinces"  # only memory, found by meaning

    store.delete(1)
    assert contents(engine) == []
    assert run(store.search("quinces")) == []


def test_unknown_memories_raise(store):
    with pytest.raises(MemoryStoreError):
        store.delete(99)
    with pytest.raises(MemoryStoreError):
        run(store.update(99, "x"))


def test_new_unrelated_fact_skips_the_llm(store, llm):
    run(store.create_memories(["Aidan likes apples"]))
    run(store.create_memories(["The deploy server is in London"]))

    assert llm.calls == []


def test_duplicate_is_skipped(engine, store, llm):
    run(store.create_memories(["Aidan likes apples"]))
    llm.decisions.append(decision("skip"))

    [report] = run(store.create_memories(["Aidan likes apples a lot"]))

    assert report.startswith("Skipped")
    assert contents(engine) == ["Aidan likes apples"]


def test_update_rewrites_the_old_memory(engine, store, llm):
    run(store.create_memories(["Aidan likes apples"]))
    llm.decisions.append(decision("update", memory_id=1, content="Aidan likes apples and pears"))

    run(store.create_memories(["Aidan also likes pears apples"]))

    assert contents(engine) == ["Aidan likes apples and pears"]
    assert run(store.search("pears"))[0].content == "Aidan likes apples and pears"


def test_outdated_memories_are_deleted_when_adding(engine, store, llm):
    run(store.create_memories(["Aidan lives in Leeds"]))
    llm.decisions.append(decision("add", delete_ids=[1]))

    [report] = run(store.create_memories(["Aidan lives in Bristol now"]))

    assert contents(engine) == ["Aidan lives in Bristol now"]
    assert "deleted" in report


def test_llm_cant_touch_memories_it_wasnt_shown(engine, store, llm):
    run(store.create_memories(["Aidan likes apples", "The deploy server is in London"]))
    llm.decisions.append(decision("add", delete_ids=[2]))

    run(store.create_memories(["Aidan likes apples and pears"]))

    assert "The deploy server is in London" in contents(engine)


def test_no_decision_adds_the_fact(engine, store):
    run(store.create_memories(["Aidan likes apples"]))

    run(store.create_memories(["Aidan likes apples a lot"]))  # the LLM makes no tool call

    assert contents(engine) == ["Aidan likes apples", "Aidan likes apples a lot"]


def test_facts_from_one_source_share_it(engine, store):
    message = chat.send_message(engine, "user", "remember some things")

    run(store.create_memories(["Aidan likes apples"], SourceRef(message_id=message.id)))
    run(store.create_memories(["The deploy server is in London"], SourceRef(message_id=message.id)))

    with Session(engine) as session:
        assert session.query(Source).count() == 1
        assert {m.source.message_id for m in session.query(Memory)} == {message.id}


def test_attachment_source_shows_in_results(engine, store):
    pdf = attachments.save(engine, "report.pdf", "application/pdf", b"%PDF")

    run(store.create_memories(["The report says revenue doubled"], SourceRef(attachment_id=pdf.id)))

    assert f"from report.pdf (attachment #{pdf.id})" in describe(run(store.search("revenue"))[0])


def test_unknown_source_is_an_error_and_saves_nothing(engine, store):
    with pytest.raises(MemoryStoreError):
        run(store.create_memories(["x"], SourceRef(attachment_id=99)))
    with pytest.raises(MemoryStoreError):
        run(store.create_memories(["x"], SourceRef(message_id=99)))

    assert contents(engine) == []


def test_source_goes_with_its_message_but_the_memory_stays(engine, store):
    message = chat.send_message(engine, "user", "remember this")
    run(store.create_memories(["Aidan likes apples"], SourceRef(message_id=message.id)))

    with Session(engine) as session:
        session.delete(session.get(ChatMessage, message.id))
        session.commit()
        [memory] = session.query(Memory).all()
        assert memory.source_id is None


def orchestrator_tools(engine, store, owner_id):
    async def on_turn_end(conversation_id: int) -> None:
        pass

    return ToolRunner(
        engine, on_turn_end, start_routine=lambda *_: None, projects_dir=Path("."),
        owner_id=owner_id, placement=Placement(parent_id=None), memory=store,
    ).tools()


def test_orchestrator_cites_the_message_that_started_its_turn(engine, store):
    message = chat.send_message(engine, "user", "I like apples")
    tools = orchestrator_tools(engine, store, owner_id=message.id)

    created = run(run_tool(tools, ToolCall(id="1", name="create_memories", arguments={"facts": ["Aidan likes apples"]})))
    found = run(run_tool(tools, ToolCall(id="2", name="search_memories", arguments={"query": "apples"})))

    assert created.startswith("Added")
    assert f"from chat message #{message.id}" in found


def test_orchestrator_has_no_memory_tools_without_a_store(engine):
    assert "search_memories" not in [t.name for t in orchestrator_tools(engine, None, owner_id=None)]


def test_mcp_server_lets_workers_search_save_and_update_but_not_delete(engine, store):
    server = build_server(store)

    listed = run(server.request_handlers[types.ListToolsRequest](types.ListToolsRequest(method="tools/list")))
    assert [t.name for t in listed.root.tools] == ["search_memories", "create_memories", "update_memory"]

    pdf = attachments.save(engine, "notes.pdf", "application/pdf", b"%PDF")
    call = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(
            name="create_memories",
            arguments={"facts": ["The plan is to ship in May"], "attachment_id": pdf.id, "message_id": None},
        ),
    )
    result = run(server.request_handlers[types.CallToolRequest](call))

    assert result.root.content[0].text.startswith("Added")
    with Session(engine) as session:
        assert session.query(Memory).one().source.attachment_id == pdf.id
