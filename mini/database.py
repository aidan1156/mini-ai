from sqlalchemy import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass



class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(unique=True)
    brief_description: Mapped[str]
    full_description: Mapped[str]

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation: Mapped[str]


def create_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)