from pathlib import Path

import pytest

from mini import people
from mini.database import init_db
from mini.orchestrator import chat, context


@pytest.fixture
def engine(tmp_path: Path):
    return init_db(tmp_path / "test.db")


@pytest.fixture
def sam(engine):
    person = people.create_person(engine, "Sam")
    people.link_identity(engine, person.id, "discord", "111")
    return person


def test_sender_is_saved_and_shown(engine, sam):
    message = chat.send_message(engine, "user", "fix the flaky test", sender=("discord", "111"))

    assert message.sender_id == sam.id
    assert context.render(message).content == f"[#{message.id} from Sam] fix the flaky test"


def test_accounts_on_different_platforms_are_the_same_person(engine, sam):
    people.link_identity(engine, sam.id, "web", "sam")
    a = chat.send_message(engine, "user", "from discord", sender=("discord", "111"))
    b = chat.send_message(engine, "user", "from the web", sender=("web", "sam"))

    assert a.sender_id == b.sender_id == sam.id


def test_unknown_account_is_rejected_and_nothing_saved(engine, sam):
    with pytest.raises(people.UnknownIdentityError):
        chat.send_message(engine, "user", "hi", sender=("discord", "999"))
    # same external id on another platform isn't the same account
    with pytest.raises(people.UnknownIdentityError):
        chat.send_message(engine, "user", "hi", sender=("web", "111"))

    assert context.chat_history(engine, None, max_tokens=1000) == []


def test_account_can_only_be_linked_once(engine, sam):
    other = people.create_person(engine, "Alex")
    with pytest.raises(ValueError):
        people.link_identity(engine, other.id, "discord", "111")


def test_renaming_a_person_changes_it_everywhere(engine, sam):
    from sqlalchemy.orm import Session
    from mini.database.models import Person

    old = chat.send_message(engine, "user", "first", sender=("discord", "111"))
    with Session(engine) as session:
        session.get(Person, sam.id).name = "Samuel"
        session.commit()

    history = [m.content for m in context.chat_history(engine, None, max_tokens=1000)]
    assert history == [f"[#{old.id} from Samuel] first"]


def test_messages_without_a_sender_are_unchanged(engine):
    message = chat.send_message(engine, "user", "hello")
    reply = chat.send_message(engine, "assistant", "hi", parent_id=message.id)

    assert message.sender is None
    assert context.render(message).content == f"[#{message.id}] hello"
    assert context.render(reply).content == "hi"


def test_search_results_name_the_sender(engine, sam):
    message = chat.send_message(engine, "user", "deploy it", sender=("discord", "111"))

    assert context.describe_match(message) == f"[#{message.id}] Sam: deploy it"
