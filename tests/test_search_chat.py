import asyncio
from pathlib import Path

import pytest

from mini.database import init_db
from mini.llm import ToolCall
from mini.orchestrator import chat, context
from mini.orchestrator.owners import Placement
from mini.orchestrator.tools import MAX_CHAT_SEARCH_LIMIT, ToolRunner, run_tool


@pytest.fixture
def engine(tmp_path: Path):
    return init_db(tmp_path / "test.db")


def runner(engine, tmp_path: Path) -> ToolRunner:
    async def on_turn_end(conversation_id: int) -> None:
        pass

    return ToolRunner(
        engine, on_turn_end, start_routine=lambda *_: None,
        projects_dir=tmp_path, owner_id=None, placement=Placement(parent_id=None),
    )


def search(engine, tmp_path: Path, **arguments) -> str:
    tool_runner = runner(engine, tmp_path)
    return asyncio.run(run_tool(tool_runner.tools(), ToolCall(id="1", name="search_chat", arguments=arguments)))


def test_finds_user_and_assistant_messages_newest_first(engine):
    question = chat.send_message(engine, "user", "Where is the Auth refactor at?")
    chat.send_message(engine, "assistant", "Unrelated")
    answer = chat.send_message(engine, "assistant", "The auth refactor is half done.", parent_id=question.id)

    found = context.search_chat(engine, "auth REFACTOR", limit=10)

    assert [m.id for m in found] == [answer.id, question.id]


def test_keeps_only_the_most_recent_matches(engine):
    ids = [chat.send_message(engine, "user", f"deploy {i}").id for i in range(5)]

    found = context.search_chat(engine, "deploy", limit=2)

    assert [m.id for m in found] == [ids[4], ids[3]]


def test_matches_wildcards_literally(engine):
    chat.send_message(engine, "user", "100 percent")
    literal = chat.send_message(engine, "user", "100% sure")

    assert [m.id for m in context.search_chat(engine, "100%", limit=10)] == [literal.id]
    assert context.search_chat(engine, "1_0", limit=10) == []


def test_tool_labels_results_with_id_thread_and_sender(engine, tmp_path):
    question = chat.send_message(engine, "user", "is the flaky test fixed?")
    answer = chat.send_message(engine, "assistant", "Not yet, the flaky test still fails.", parent_id=question.id)

    result = search(engine, tmp_path, text="flaky test")

    assert result == (
        f"[#{answer.id} in thread #{question.id}] you: Not yet, the flaky test still fails.\n\n"
        f"[#{question.id}] user: is the flaky test fixed?"
    )


def test_tool_limit_defaults_and_is_capped(engine, tmp_path):
    for i in range(MAX_CHAT_SEARCH_LIMIT + 5):
        chat.send_message(engine, "user", f"note {i}")

    assert search(engine, tmp_path, text="note", limit=3).count("user: note") == 3
    assert search(engine, tmp_path, text="note").count("user: note") == 10
    assert search(engine, tmp_path, text="note", limit=1000).count("user: note") == MAX_CHAT_SEARCH_LIMIT
    assert search(engine, tmp_path, text="note", limit=0).count("user: note") == 10


def test_tool_reports_no_matches_and_empty_text(engine, tmp_path):
    chat.send_message(engine, "user", "hello")

    assert search(engine, tmp_path, text="goodbye") == "No matches."
    assert search(engine, tmp_path, text="  ").startswith("Error:")
