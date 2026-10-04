import discord
from discord import app_commands

from . import llm, memory
from .client import bot
from .formatting import split_chunks
from .settings import settings


@bot.tree.command(name="sakiya", description="Summon sakiya anywhere as an app")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.describe(message="What do you want to say?")
async def sakiya(interaction: discord.Interaction, message: str):
    await interaction.response.defer(thinking=True)
    try:
        reply_text = await llm.generate_bot_reply(interaction.channel, interaction.user, message.strip())

        chunks = split_chunks(reply_text)

        await interaction.followup.send(chunks[0])
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)
    except Exception as e:
        print("ERROR:", e)
        await interaction.followup.send("Error communicating with local model.")


@bot.tree.command(name="say", description="Make sakiya repeat a message word for word")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.describe(message="The exact text sakiya should say")
async def say(interaction: discord.Interaction, message: str):
    await interaction.response.defer(thinking=True)

    text = message.strip()
    if not text:
        await interaction.followup.send("_ _")
        return

    chunks = split_chunks(text)

    await interaction.followup.send(chunks[0])
    for chunk in chunks[1:]:
        await interaction.followup.send(chunk)


@bot.tree.command(name="sync_memory", description="Absorb recent messages in this channel into channel memory")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def sync_memory(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    try:
        messages = [m async for m in interaction.channel.history(limit=settings.max_channel_memory)]
        messages.reverse()

        channel_id = interaction.channel.id
        entries: list[dict[str, str]] = []

        for msg in messages:
            if not msg.content or msg.content.startswith("/"):
                continue
            if msg.author.id == bot.user.id:
                entries.append({"role": "assistant", "content": msg.content})
            else:
                entries.append({"role": "user", "content": f"{msg.author.display_name}: {msg.content}"})

        memory.replace_channel_history(channel_id, entries)

        await interaction.followup.send(f"✅ Absorbed the last {len(entries)} messages for this chat.", ephemeral=True)
    except discord.Forbidden:
        await interaction.followup.send("❌ Cannot read history in this channel/DM due to permissions.", ephemeral=True)
    except Exception as e:
        print("ERROR:", e)
        await interaction.followup.send("❌ Something went wrong reading chat history.", ephemeral=True)


@bot.tree.command(name="clear_memory", description="Wipes current user's channel and cross-channel memory with sakiya")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def clear_memory(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    channel_id = interaction.channel.id
    user_id = interaction.user.id

    memory.clear_channel_memory(channel_id)
    memory.clear_user_memory(user_id)

    await interaction.followup.send("🧠 Channel memory and global user context wiped! Clean slate.", ephemeral=True)
