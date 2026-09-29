"""A Discord bot that is a client of mini's API.

- A message in the home channel (or a thread off it) is POSTed to the API, and
  linked to the chat message it became (the `discord_messages` table).
- Mini's messages arrive on the API's event stream. A top-level one is posted
  in the home channel; a reply goes in the thread off its parent's Discord
  message, starting the thread if there isn't one yet.

User messages on the event stream are skipped: the ones sent from Discord are
already there, and ones sent from other clients aren't mirrored.

Files go both ways: ones attached in Discord are uploaded to mini with the
message, and ones on mini's messages are posted as Discord attachments.
"""

import asyncio
import io
import logging

import aiohttp
import discord
from sqlalchemy import Engine, func, select, update
from sqlalchemy.orm import Session

from mini.api.public_models import ChatMessageOut
from mini.attachments import MAX_SIZE
from mini.database.models import ChatMessage, DiscordMessage
from mini.discord.api_client import MiniClient

logger = logging.getLogger(__name__)

DISCORD_MESSAGE_LIMIT = 2000
THREAD_NAME_LIMIT = 100
RECONNECT_SECONDS = 5
# A reply's parent is linked just after the API returns it, which can race with
# the reply arriving on the event stream. Wait this long for the link to appear.
PARENT_LINK_WAIT_SECONDS = 5
# Show "typing" after the user's message until mini posts anything, or this long.
TYPING_TIMEOUT_SECONDS = 60
FILES_PER_MESSAGE = 10  # Discord's limit


class MiniBot(discord.Client):
    def __init__(self, engine: Engine, api: MiniClient, channel_id: int):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.engine = engine
        self.api = api
        self.channel_id = channel_id
        # Set (and replaced with a fresh one) whenever mini posts, to stop every typing indicator.
        self._posted = asyncio.Event()
        self._typing_tasks: set[asyncio.Task] = set()

    async def setup_hook(self) -> None:
        self.loop.create_task(self._follow_events())

    async def on_ready(self) -> None:
        logger.info("Discord bot online as %s", self.user)

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or not (message.content or message.attachments):
            return
        # Skip Discord's system messages, e.g. the "started a thread" notice (whose text
        # is the thread's name).
        if message.type not in (discord.MessageType.default, discord.MessageType.reply):
            return

        channel = message.channel
        if isinstance(channel, discord.Thread):
            if channel.parent_id != self.channel_id:
                return
            parent_id = self._chat_id_for_thread(channel.id)
            if parent_id is None:
                return  # a thread mini doesn't know about
        elif channel.id == self.channel_id:
            parent_id = None
        else:
            return

        try:
            attachment_ids = await self._upload_attachments(message)
            if not message.content and not attachment_ids:
                return  # only files, and none could be sent (already flagged)
            sent = await self.api.send_message(message.content, parent_id, attachment_ids)
        except aiohttp.ClientError:
            logger.exception("Couldn't send message to mini")
            await message.add_reaction("⚠️")
            return
        self._link(sent.id, message)

        task = asyncio.create_task(self._type_until_posted(channel))
        self._typing_tasks.add(task)
        task.add_done_callback(self._typing_tasks.discard)

    async def _upload_attachments(self, message: discord.Message) -> list[int]:
        """Upload the message's files to mini, returning their ids. Too-big ones get a ⚠️."""
        ids = []
        for attachment in message.attachments:
            if attachment.size > MAX_SIZE:
                logger.warning("Skipping %s: %s bytes is over mini's limit", attachment.filename, attachment.size)
                await message.add_reaction("⚠️")
                continue
            uploaded = await self.api.upload_attachment(
                attachment.filename, attachment.content_type, await attachment.read()
            )
            ids.append(uploaded.id)
        return ids

    async def _type_until_posted(self, channel: discord.abc.Messageable) -> None:
        posted = self._posted
        async with channel.typing():
            try:
                await asyncio.wait_for(posted.wait(), TYPING_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                pass

    async def _follow_events(self) -> None:
        """Post mini's messages to Discord, reconnecting to the event stream as needed."""
        await self.wait_until_ready()
        while not self.is_closed():
            try:
                async for event in self.api.events(after=self._last_linked_id()):
                    if event.role == "assistant":
                        await self._post(event)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning("Event stream disconnected (%s); reconnecting", e)
            except Exception:
                logger.exception("Event stream failed; reconnecting")
            await asyncio.sleep(RECONNECT_SECONDS)

    async def _post(self, event: ChatMessageOut) -> None:
        if self._get_link(event.id) is not None:
            return  # already posted (replayed after a reconnect)

        if event.parent_id is None:
            channel = self.get_channel(self.channel_id) or await self.fetch_channel(self.channel_id)
        else:
            channel = await self._thread_for(event.parent_id)

        files, too_big = await self._download_attachments(event, channel)
        content = event.content
        if too_big:
            content += f"\n\n(Too big to post on Discord: {', '.join(too_big)})"

        # The text (split to fit), with the files on the last part, 10 to a message.
        sends: list[dict] = [{"content": chunk} for chunk in _split(content)] if content.strip() else []
        batches = [files[i:i + FILES_PER_MESSAGE] for i in range(0, len(files), FILES_PER_MESSAGE)]
        if batches and sends:
            sends[-1]["files"] = batches.pop(0)
        sends += [{"files": batch} for batch in batches]
        if not sends:
            return  # nothing that can be shown

        first = await channel.send(**sends[0])
        for kwargs in sends[1:]:
            await channel.send(**kwargs)
        self._link(event.id, first)

        self._posted.set()
        self._posted = asyncio.Event()

    async def _download_attachments(
        self, event: ChatMessageOut, channel: discord.abc.Messageable
    ) -> tuple[list[discord.File], list[str]]:
        """The event's files as Discord files, and the names of any too big to upload here."""
        guild = getattr(channel, "guild", None)
        limit = guild.filesize_limit if guild is not None else 10 * 1024 * 1024
        files, too_big = [], []
        for attachment in event.attachments:
            if attachment.size > limit:
                too_big.append(attachment.filename)
                continue
            data = await self.api.download_attachment(attachment.id)
            files.append(discord.File(io.BytesIO(data), filename=attachment.filename))
        return files, too_big

    async def _thread_for(self, parent_id: int) -> discord.abc.Messageable:
        """The thread off the parent's Discord message, started if needed."""
        link = await self._wait_for_link(parent_id)
        if link is None:
            logger.warning("Chat message %s isn't on Discord; posting its reply top-level", parent_id)
            return self.get_channel(self.channel_id) or await self.fetch_channel(self.channel_id)

        if link.thread_id is not None:
            return self.get_channel(link.thread_id) or await self.fetch_channel(link.thread_id)

        channel = self.get_channel(link.channel_id) or await self.fetch_channel(link.channel_id)
        parent_message = await channel.fetch_message(link.discord_message_id)
        thread = parent_message.thread or await parent_message.create_thread(
            name=_thread_name(self._chat_content(parent_id)), auto_archive_duration=1440
        )
        with Session(self.engine) as session:
            session.execute(
                update(DiscordMessage)
                .where(DiscordMessage.chat_message_id == parent_id)
                .values(thread_id=thread.id)
            )
            session.commit()
        return thread

    async def _wait_for_link(self, chat_message_id: int) -> DiscordMessage | None:
        for _ in range(PARENT_LINK_WAIT_SECONDS * 4):
            if (link := self._get_link(chat_message_id)) is not None:
                return link
            await asyncio.sleep(0.25)
        return None

    def _link(self, chat_message_id: int, message: discord.Message) -> None:
        with Session(self.engine) as session:
            session.add(DiscordMessage(
                chat_message_id=chat_message_id,
                discord_message_id=message.id,
                channel_id=message.channel.id,
            ))
            session.commit()

    def _get_link(self, chat_message_id: int) -> DiscordMessage | None:
        with Session(self.engine, expire_on_commit=False) as session:
            return session.get(DiscordMessage, chat_message_id)

    def _chat_id_for_thread(self, thread_id: int) -> int | None:
        """The chat message a thread belongs to, or None if it isn't one of mini's."""
        with Session(self.engine) as session:
            chat_message_id = session.scalar(
                select(DiscordMessage.chat_message_id).where(DiscordMessage.thread_id == thread_id)
            )
            if chat_message_id is not None:
                return chat_message_id

            # A thread the user started off a message has the same id as that message.
            link = session.scalar(
                select(DiscordMessage).where(DiscordMessage.discord_message_id == thread_id)
            )
            if link is None:
                return None
            link.thread_id = thread_id  # so mini's replies go in this thread too
            session.commit()
            return link.chat_message_id

    def _chat_content(self, chat_message_id: int) -> str:
        with Session(self.engine) as session:
            return session.get(ChatMessage, chat_message_id).content

    def _last_linked_id(self) -> int | None:
        """Where to resume the event stream: everything after the last message on Discord."""
        with Session(self.engine) as session:
            return session.scalar(select(func.max(DiscordMessage.chat_message_id)))


def _split(content: str) -> list[str]:
    """Split a message into Discord-sized chunks, preferring line breaks."""
    chunks = []
    while len(content) > DISCORD_MESSAGE_LIMIT:
        cut = content.rfind("\n", 0, DISCORD_MESSAGE_LIMIT)
        if cut <= 0:
            cut = DISCORD_MESSAGE_LIMIT
        chunks.append(content[:cut])
        content = content[cut:].lstrip("\n")
    chunks.append(content)
    return chunks


def _thread_name(content: str) -> str:
    first_line = content.strip().splitlines()[0] if content.strip() else "mini"
    if len(first_line) <= THREAD_NAME_LIMIT:
        return first_line
    return first_line[: THREAD_NAME_LIMIT - 1] + "…"
