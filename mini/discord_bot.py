from pathlib import Path

import discord
from dotenv import load_dotenv
import os
import asyncio
import aiohttp

from mini.config import load_config
from mini.public_models import SendMessageResponse

load_dotenv()

repo_root = Path(__file__).resolve().parent.parent
config = load_config(str(repo_root / "config.json"))

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready() -> None:
    print(f"Bot is online as {client.user}")

@client.event
async def on_message(message: discord.Message) -> None:
    print(f"Received message: '{message.content}' from {message.author}")
    if message.author == client.user:
        return

    user_message = message.content

    conversation_id = None
    if isinstance(message.channel, discord.Thread):
        conversation_id = message.channel.name.split("-")[-1]

    response = await send_message(user_message, conversation_id=conversation_id)

    if conversation_id is None:
        channel = await message.create_thread(name=f"chat-{response.conversation_id}", auto_archive_duration=1440)
    else:
        channel = message.channel

    await channel.send(response.response)

async def send_message(message: str, conversation_id: int | None = None) -> SendMessageResponse:
    url = f"http://{config.server.host}:{config.server.port}/send-message"
    payload = {"message": message, "conversation_id": conversation_id}
    headers = {"Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            response_json = await response.json()

            response_model = SendMessageResponse.model_validate(response_json)
            return response_model
                

client.run(os.environ["DISCORD_BOT_TOKEN"])