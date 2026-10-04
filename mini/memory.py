"""The orchestrator's memory: small atomic facts, searched by meaning and by words.

Used by both the orchestrator's tools and the workers' MCP server (see
`memory_tools`), so there's one implementation.

A search runs two lookups over the memories -- the nearest embeddings to the
query (sqlite-vec, brute force), and a full-text match so odd, project-specific
words that embeddings blur still hit -- and merges the two rankings.

Saving a fact first checks the memories nearest to it: a small LLM call decides
whether it's new, a duplicate, or an update to what's already there (and which
existing memories it makes out of date), so the memory doesn't fill with
repeats or stale facts.
"""

import logging
import re
from dataclasses import dataclass, field

import sqlite_vec
from sqlalchemy import Engine, delete, select, text
from sqlalchemy.orm import Session

from mini.database.models import Attachment, ChatMessage, Memory, Source
from mini.llm import LLM, Embedder, Message, Tool

logger = logging.getLogger(__name__)

SEARCH_LIMIT = 20  # results per search, and candidates from each of the two lookups
# A new fact is only compared to memories this close (cosine distance; 0 is identical).
SIMILAR_MAX_DISTANCE = 0.6
SIMILAR_LIMIT = 5

CONSOLIDATE_PROMPT = """\
You keep an assistant's memory of small, atomic facts tidy. A new fact is about
to be saved, and you're shown the existing memories most similar to it. Call
decide with one action:
- add: the new fact is new information. Also list in delete_ids any existing
  memories it makes out of date or contradicts.
- skip: an existing memory already says the same thing.
- update: an existing memory covers the same thing but the new fact refines or
  corrects it. Set memory_id to that memory and content to the combined, correct
  fact, written as one self-contained statement.
Only refer to the listed memories. When unsure, add."""


class MemoryStoreError(Exception):
    """Something the caller asked for can't be done; the message says why."""


@dataclass(frozen=True)
class SourceRef:
    """Where a fact came from. If both are set, the attachment is the source."""

    attachment_id: int | None = None
    message_id: int | None = None


@dataclass
class _Decision:
    action: str = "add"  # add | skip | update
    memory_id: int | None = None
    content: str | None = None
    delete_ids: list[int] = field(default_factory=list)


class MemoryStore:
    """Save, change, delete and search memories. `llm` decides how new facts fit with old ones."""

    def __init__(self, engine: Engine, embedder: Embedder, llm: LLM):
        self.engine = engine
        self.embedder = embedder
        self.llm = llm

    async def create_memories(self, facts: list[str], source: SourceRef | None = None) -> list[str]:
        """Save facts, one at a time so later ones see earlier ones. Returns what happened to each."""
        facts = [fact.strip() for fact in facts if fact.strip()]
        if not facts:
            raise MemoryStoreError("no facts given")
        embeddings = await self._embed(facts)
        source_id = self._get_or_create_source(source)
        return [await self._create_one(fact, embedding, source_id) for fact, embedding in zip(facts, embeddings)]

    async def update(self, memory_id: int, content: str) -> Memory:
        """Replace a memory's content, keeping its source."""
        content = content.strip()
        if not content:
            raise MemoryStoreError("content is empty")
        [embedding] = await self._embed([content])
        with Session(self.engine, expire_on_commit=False) as session:
            memory = session.get(Memory, memory_id)
            if memory is None:
                raise MemoryStoreError(f"no memory #{memory_id}")
            memory.content = content
            memory.embedding = embedding
            memory.embedding_model = self.embedder.embedding_model
            session.commit()
            return memory

    def delete(self, memory_id: int) -> None:
        with Session(self.engine) as session:
            if session.execute(delete(Memory).where(Memory.id == memory_id)).rowcount == 0:
                raise MemoryStoreError(f"no memory #{memory_id}")
            session.commit()

    async def search(self, query: str) -> list[Memory]:
        """Up to SEARCH_LIMIT memories matching `query` by meaning or by words, best first."""
        print("searching for", query)
        [embedding] = await self._embed([query])
        with Session(self.engine, expire_on_commit=False) as session:
            by_meaning = [id_ for id_, _ in self._nearest(session, embedding, SEARCH_LIMIT)]
            by_words = self._matching_words(session, query)

            # Merge the rankings (reciprocal rank fusion): what ranks well in either comes first.
            scores: dict[int, float] = {}
            for ranking in (by_meaning, by_words):
                for rank, memory_id in enumerate(ranking):
                    scores[memory_id] = scores.get(memory_id, 0) + 1 / (60 + rank)
            ids = sorted(scores, key=lambda id_: -scores[id_])[:SEARCH_LIMIT]
            memories = {m.id: m for m in session.scalars(select(Memory).where(Memory.id.in_(ids)))}
        return [memories[id_] for id_ in ids]

    async def _create_one(self, fact: str, embedding: bytes, source_id: int | None) -> str:
        with Session(self.engine, expire_on_commit=False) as session:
            similar = [
                session.get(Memory, id_)
                for id_, distance in self._nearest(session, embedding, SIMILAR_LIMIT)
                if distance <= SIMILAR_MAX_DISTANCE
            ]
        decision = await self._decide(fact, similar) if similar else _Decision()
        similar_ids = {m.id for m in similar}

        if decision.action == "skip":
            return f"Skipped, already known: {fact}"

        if decision.action == "update" and decision.memory_id in similar_ids:
            content = (decision.content or fact).strip()
            if content != fact:
                [embedding] = await self._embed([content])
            with Session(self.engine) as session:
                memory = session.get(Memory, decision.memory_id)
                logger.info("Memory #%s updated from %r to %r", memory.id, memory.content, content)
                memory.content = content
                memory.embedding = embedding
                memory.embedding_model = self.embedder.embedding_model
                if source_id is not None:
                    memory.source_id = source_id
                session.commit()
            return f"Updated #{decision.memory_id}: {content}"

        with Session(self.engine) as session:
            memory = Memory(
                content=fact, embedding=embedding, embedding_model=self.embedder.embedding_model,
                source_id=source_id,
            )
            session.add(memory)
            deleted = []
            for id_ in set(decision.delete_ids) & similar_ids:
                old = session.get(Memory, id_)
                logger.info("Memory #%s deleted as out of date: %r", old.id, old.content)
                deleted.append(f"#{old.id} ({old.content})")
                session.delete(old)
            session.commit()
            report = f"Added #{memory.id}: {fact}"
        return f"{report}; deleted out-of-date {', '.join(deleted)}" if deleted else report

    async def _decide(self, fact: str, similar: list[Memory]) -> _Decision:
        async def noop(**_) -> str:
            return "Done."

        tool = Tool(
            name="decide",
            description="Say what to do with the new fact.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "skip", "update"]},
                    "memory_id": {"type": ["integer", "null"], "description": "For update: the memory to rewrite."},
                    "content": {"type": ["string", "null"], "description": "For update: the memory's new content."},
                    "delete_ids": {
                        "type": "array", "items": {"type": "integer"},
                        "description": "For add: existing memories the new fact makes out of date.",
                    },
                },
                "required": ["action", "memory_id", "content", "delete_ids"],
            },
            handler=noop,  # not run: the arguments are the answer
        )
        listing = "\n".join(describe(m) for m in similar)
        reply = await self.llm.complete(
            CONSOLIDATE_PROMPT,
            [Message(role="user", content=f"New fact: {fact}\n\nSimilar existing memories:\n{listing}")],
            [tool],
        )
        call = next((c for c in reply.tool_calls if c.name == tool.name), None)
        if call is None:
            logger.warning("Memory consolidation made no decision; adding the fact")
            return _Decision()
        arguments = call.arguments
        return _Decision(
            action=arguments.get("action") if arguments.get("action") in ("add", "skip", "update") else "add",
            memory_id=arguments.get("memory_id"),
            content=arguments.get("content"),
            delete_ids=[i for i in arguments.get("delete_ids") or [] if isinstance(i, int)],
        )

    def _nearest(self, session: Session, embedding: bytes, limit: int) -> list[tuple[int, float]]:
        """(id, cosine distance) of the memories nearest to `embedding`, closest first."""
        rows = session.execute(
            text(
                "SELECT id, vec_distance_cosine(embedding, :query) AS distance FROM memories "
                "WHERE embedding_model = :model ORDER BY distance LIMIT :limit"
            ),
            {"query": embedding, "model": self.embedder.embedding_model, "limit": limit},
        )
        return [(row.id, row.distance) for row in rows]

    @staticmethod
    def _matching_words(session: Session, query: str) -> list[int]:
        # Quote each word so FTS5 syntax in the query can't break the search; any word may match.
        words = re.findall(r"\w+", query)
        if not words:
            return []
        match = " OR ".join(f'"{word}"' for word in words)
        rows = session.execute(
            text("SELECT rowid FROM memories_fts WHERE memories_fts MATCH :match ORDER BY rank LIMIT :limit"),
            {"match": match, "limit": SEARCH_LIMIT},
        )
        return [row[0] for row in rows]

    def _get_or_create_source(self, ref: SourceRef | None) -> int | None:
        if ref is None or (ref.attachment_id is None and ref.message_id is None):
            return None
        with Session(self.engine) as session:
            if ref.attachment_id is not None:
                if session.get(Attachment, ref.attachment_id) is None:
                    raise MemoryStoreError(f"no attachment #{ref.attachment_id}")
                where = Source.attachment_id == ref.attachment_id
                new = Source(attachment_id=ref.attachment_id)
            else:
                if session.get(ChatMessage, ref.message_id) is None:
                    raise MemoryStoreError(f"no chat message #{ref.message_id}")
                where = Source.message_id == ref.message_id
                new = Source(message_id=ref.message_id)
            source = session.scalar(select(Source).where(where))
            if source is None:
                session.add(new)
                session.commit()
                source = new
            return source.id

    async def _embed(self, texts: list[str]) -> list[bytes]:
        return [sqlite_vec.serialize_float32(vector) for vector in await self.embedder.embed(texts)]


def describe(memory: Memory) -> str:
    """One line for the model: id, date, source and content."""
    where = ""
    if source := memory.source:
        if source.attachment is not None:
            where = f", from {source.attachment.filename} (attachment #{source.attachment_id})"
        elif source.message_id is not None:
            where = f", from chat message #{source.message_id}"
    return f"#{memory.id} [{memory.updated_at:%Y-%m-%d}{where}] {memory.content}"
