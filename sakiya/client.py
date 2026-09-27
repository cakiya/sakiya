import discord
from discord.ext import commands

from . import llm
from .formatting import split_chunks
from .settings import settings

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=settings.command_prefix, intents=intents)


@bot.event
async def on_ready():
    print("Syncing slash commands globally...")
    await bot.tree.sync()
    print(f"Logged in as {bot.user} | Slash commands synced.")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user in message.mentions

    if is_dm or is_mentioned:
        user_input = message.content.replace(f'<@{bot.user.id}>', '').strip()
        if not user_input:
            return

        async with message.channel.typing():
            try:
                reply_text = await llm.generate_bot_reply(message.channel, message.author, user_input)

                for chunk in split_chunks(reply_text):
                    await message.channel.send(chunk)
            except Exception as e:
                print("ERROR:", e)
                await message.channel.send("Error communicating with local model.")


from . import commands as _commands  # noqa: E402,F401  registers the slash commands
