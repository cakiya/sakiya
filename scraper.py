import os
import discord
from dotenv import load_dotenv

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

TARGET_USER_ID = int(os.getenv("TARGET_USER_ID", "0"))
TARGET_GUILD_ID = int(os.getenv("TARGET_GUILD_ID", "0"))
OUTPUT_FILE = "scraped_messages.txt"

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

    guild = client.get_guild(TARGET_GUILD_ID)
    if not guild:
        print(f"Could not find guild with ID {TARGET_GUILD_ID}")
        await client.close()
        return

    user = await client.fetch_user(TARGET_USER_ID)
    if not user:
        print(f"Could not find user with ID {TARGET_USER_ID}")
        await client.close()
        return

    messages = []
    for channel in guild.text_channels:
        try:
            async for message in channel.history(limit=None):
                if message.author.id == TARGET_USER_ID:
                    messages.append(f"{message.content}")
        except discord.Forbidden:
            continue

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(messages))

    print(f"Scraped {len(messages)} messages from {user.name} to {OUTPUT_FILE}")
    await client.close()

client.run(DISCORD_TOKEN)
