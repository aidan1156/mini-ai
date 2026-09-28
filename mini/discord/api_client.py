"""A small client for mini's HTTP API (see mini/api/app.py)."""

import json
from collections.abc import AsyncIterator

import aiohttp

from mini.api.public_models import ChatMessageOut, SendMessageIn

# The server pings every 15s, so a minute of silence means the connection is dead.
EVENTS_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_read=60)


class MiniClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "MiniClient":
        self.session = aiohttp.ClientSession(self.base_url, headers=self.headers)
        return self

    async def __aexit__(self, *exc) -> None:
        await self.session.close()

    async def send_message(self, content: str, parent_id: int | None = None) -> ChatMessageOut:
        body = SendMessageIn(content=content, parent_id=parent_id)
        async with self.session.post("/messages", json=body.model_dump()) as response:
            response.raise_for_status()
            return ChatMessageOut.model_validate(await response.json())

    async def events(self, after: int | None = None) -> AsyncIterator[ChatMessageOut]:
        """Yield chat messages from the event stream, starting after message `after`.

        Runs until the connection drops; reconnect by calling again with the last id seen.
        """
        params = {"after": after} if after is not None else {}
        async with self.session.get("/events", params=params, timeout=EVENTS_TIMEOUT) as response:
            response.raise_for_status()
            data: list[str] = []
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data.append(line[5:].removeprefix(" "))
                elif not line and data:  # a blank line ends the event
                    yield ChatMessageOut.model_validate(json.loads("\n".join(data)))
                    data = []
                # ignore id:, event:, retry: and ": ping" comments
