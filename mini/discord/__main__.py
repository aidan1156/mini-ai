"""Run the Discord bot: python -m mini.discord

Needs the API server running (python -m mini.api), and in .env:
DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID (the channel mini lives in),
MINI_API_TOKEN, and optionally MINI_API_URL (default http://127.0.0.1:8000).
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from mini.database import init_db
from mini.discord.api_client import MiniClient
from mini.discord.bot import MiniBot


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    async with MiniClient(
        os.environ.get("MINI_API_URL", "http://127.0.0.1:8000"),
        os.environ["MINI_API_TOKEN"],
    ) as api:
        bot = MiniBot(init_db(), api, int(os.environ["DISCORD_CHANNEL_ID"]))
        async with bot:
            await bot.start(os.environ["DISCORD_BOT_TOKEN"])


asyncio.run(main())
