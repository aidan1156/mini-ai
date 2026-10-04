"""The people who send messages, and the platform accounts that map to them.

People are added by hand (see `main`); a message from an account that isn't
linked to anyone is rejected rather than creating someone.

    python -m mini.people add Aidan
    python -m mini.people link 1 discord 123456789
    python -m mini.people link 1 web aidan
"""

import argparse

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from mini.database import init_db
from mini.database.models import Person, PersonIdentity


class UnknownIdentityError(Exception):
    """A message came from a platform account that isn't linked to a person."""


def create_person(engine: Engine, name: str) -> Person:
    with Session(engine, expire_on_commit=False) as session:
        person = Person(name=name)
        session.add(person)
        session.commit()
        return person


def link_identity(engine: Engine, person_id: int, provider: str, external_id: str) -> None:
    """Link a platform account to a person. Raises if it's already linked, or the person doesn't exist."""
    with Session(engine) as session:
        if session.get(Person, person_id) is None:
            raise ValueError(f"No person {person_id}")
        if session.get(PersonIdentity, (provider, external_id)) is not None:
            raise ValueError(f"{provider}:{external_id} is already linked")
        session.add(PersonIdentity(provider=provider, external_id=external_id, person_id=person_id))
        session.commit()


def resolve(session: Session, provider: str, external_id: str) -> int:
    """The id of the person owning a platform account. Raises UnknownIdentityError if none."""
    identity = session.get(PersonIdentity, (provider, external_id))
    if identity is None:
        raise UnknownIdentityError(f"No person is linked to {provider}:{external_id}")
    return identity.person_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="add a person")
    add.add_argument("name")
    link = commands.add_parser("link", help="link a platform account to a person")
    link.add_argument("person_id", type=int)
    link.add_argument("provider")
    link.add_argument("external_id")
    args = parser.parse_args()

    engine = init_db()
    if args.command == "add":
        person = create_person(engine, args.name)
        print(f"Added {person.name} as person {person.id}")
    else:
        link_identity(engine, args.person_id, args.provider, args.external_id)
        print(f"Linked {args.provider}:{args.external_id} to person {args.person_id}")


if __name__ == "__main__":
    main()
