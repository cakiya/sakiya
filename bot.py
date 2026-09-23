import os
import re
import glob
import asyncio
import numpy as np
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
import koboldcpp
import json

# 1. Configuration
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
LLM_URL = "http://localhost:5001/v1"
PROMPT_PATH = os.path.join(BASE_DIR, "input/prompt.txt")
PERSONAS_DIR = os.path.join(BASE_DIR, "input/personas")
KOBOLD_EXE_PATH = os.path.join(BASE_DIR, "koboldcpp.exe")
MODEL_PATH = os.path.join(BASE_DIR, "input/models/" + os.getenv("MODEL_NAME", "default") + ".gguf")

# Start the background server
server_process = koboldcpp.start_server(KOBOLD_EXE_PATH, MODEL_PATH)

# Store rolling conversation history by channel
channel_memory = {}
global_user_memory = {}
MAX_CHANNEL_MEMORY = 100
MAX_USER_MEMORY = 10

# 2. Local RAG Initialization
print("Loading embedding model and knowledge files...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)

history_lines = []
history_sources = []

for filepath in glob.glob(os.path.join(PERSONAS_DIR, "*.txt")):
    filename = os.path.basename(filepath)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    history_lines.append(stripped_line)
                    history_sources.append(filename)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

if history_lines:
    history_embeddings = embedder.encode(history_lines, convert_to_numpy=True)
    print(f"Successfully embedded {len(history_lines)} lines across {len(set(history_sources))} file(s).")
else:
    print("WARNING: No text files found in the personas folder. Running without RAG context.")
    history_embeddings = np.array([])

try:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    print(f"Successfully loaded system prompt from {PROMPT_PATH}.")
except FileNotFoundError:
    print(f"WARNING: {PROMPT_PATH} not found. Using default system prompt.")
    system_prompt = "You are sakiya."

def get_relevant_context(query: str, top_k: int = 5) -> str:
    if not history_lines or top_k == 0:
        return ""
    query_vec = embedder.encode([query], convert_to_numpy=True)
    scores = np.dot(history_embeddings, query_vec.T).squeeze()
    
    # Handle single-entry array dimensions
    if scores.ndim == 0:
        scores = np.array([scores.item()])
        
    actual_k = min(top_k, len(scores))
    if actual_k == 0:
        return ""
        
    top_indices = np.argsort(scores)[::-1][:actual_k]
    
    retrieved_chunks = []
    for i in top_indices:
        retrieved_chunks.append(f"[Source: {history_sources[i]}] {history_lines[i]}")
        
    return "\n".join(retrieved_chunks)

# 3. Connect to local backend
llm_client = AsyncOpenAI(base_url=LLM_URL, api_key="koboldcpp")

# 4. Configure Discord Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

def sanitize_response(text: str) -> str:
    """Cleans up periods and stray quotes to match casual style."""
    if not text:
        return ""
    text = re.sub(r'(?i)^sakiya:\s*', '', text)
    text = text.replace('"', '')
    text = re.sub(r'\n{3,}', '\n\n', text)

    if len(text) < 160:
        text = text.replace("...", "<ELLIPSIS>")
        text = re.sub(r'\.(?!\S)', '', text)
        text = text.replace("。", "")
        text = text.replace("<ELLIPSIS>", "...")
    return text.strip()

async def generate_bot_reply(channel: discord.abc.Messageable, author: discord.User | discord.Member, user_input: str) -> str:
    channel_id = channel.id
    user_id = author.id
    
    # 1. RAG lookup
    context = await asyncio.to_thread(get_relevant_context, user_input, 5)
    
    # 2. Build human-readable channel/server context
    if isinstance(channel, discord.DMChannel):
        location_desc = f"Direct Messages with @{author.display_name}"
    elif hasattr(channel, "guild") and channel.guild:
        location_desc = f"Server: '{channel.guild.name}', Channel: #{getattr(channel, 'name', 'chat')}"
    else:
        location_desc = "Private Group Chat"

    # 3. Inject cross-channel global user memory
    user_context = ""
    if user_id in global_user_memory and global_user_memory[user_id]:
        user_context = f"\n\n[Your recent cross-channel memories with @{author.display_name}]:\n"
        for mem in global_user_memory[user_id]:
            user_context += f"- {mem}\n"

    # 4. Assemble the System Prompt
    meta_system = f"{system_prompt}\n\n[Current Chat Location: {location_desc}]{user_context}"
    if context:
        meta_system += f"\n\n[Background Knowledge Retrieved from Memory]:\n{context}"

    # 5. Manage Rolling Channel History
    if channel_id not in channel_memory:
        channel_memory[channel_id] = []

    safe_name = author.display_name
    if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', safe_name):
        safe_name = author.name if not re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', author.name) else "User"

    formatted_user_msg = f"{safe_name}: {user_input}"
    
    channel_memory[channel_id].append({"role": "user", "content": formatted_user_msg})
    if len(channel_memory[channel_id]) > MAX_CHANNEL_MEMORY:
        channel_memory[channel_id].pop(0)

    messages_payload = [{"role": "system", "content": meta_system}] + channel_memory[channel_id]

    # 6. Call LLM
    print("\n=== INCOMING LLM PAYLOAD ===")
    print(json.dumps(messages_payload, indent=2, ensure_ascii=False))
    print("============================\n")
    
    response = await llm_client.chat.completions.create(
        model="local-model",
        messages=messages_payload,
        temperature=0.85,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        extra_body={
            "min_p": 0.05,
            "top_p": 1.0,
            "rep_pen": 1.0,
            "dry_multiplier": 0.8,
            "dry_base": 1.75,
            "dry_allowed_length": 2,
            "dry_penalty_last_n": 1024
        }
    )
    
    raw_reply = response.choices[0].message.content
    clean_reply = sanitize_response(raw_reply)

    channel_memory[channel_id].append({"role": "assistant", "content": clean_reply})

    # 7. Update Global User Memory
    if user_id not in global_user_memory:
        global_user_memory[user_id] = []
        
    global_user_memory[user_id].append(f"They said: '{user_input}' | You replied: '{clean_reply}'")
    
    if len(global_user_memory[user_id]) > MAX_USER_MEMORY:
        global_user_memory[user_id].pop(0)

    return clean_reply

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
                reply_text = await generate_bot_reply(message.channel, message.author, user_input)
                
                chunk_size = 1950
                for i in range(0, len(reply_text), chunk_size):
                    await message.channel.send(reply_text[i:i+chunk_size])
            except Exception as e:
                print("ERROR:", e)
                await message.channel.send("Error communicating with local model.")

# 5. Slash Commands
@bot.tree.command(name="sakiya", description="Summon sakiya anywhere as an app")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.describe(message="What do you want to say?")
async def sakiya(interaction: discord.Interaction, message: str):
    await interaction.response.defer(thinking=True)
    try:
        reply_text = await generate_bot_reply(interaction.channel, interaction.user, message.strip())
        
        chunk_size = 1950
        chunks = [reply_text[i:i + chunk_size] for i in range(0, len(reply_text), chunk_size)]
        
        await interaction.followup.send(chunks[0])
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)
    except Exception as e:
        print("ERROR:", e)
        await interaction.followup.send("Error communicating with local model.")

@bot.tree.command(name="sync_memory", description="Absorb recent messages in this channel into channel memory")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def sync_memory(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    try:
        messages = [m async for m in interaction.channel.history(limit=MAX_USER_MEMORY)]
        messages.reverse()

        channel_id = interaction.channel.id
        channel_memory[channel_id] = []

        for msg in messages:
            if not msg.content or msg.content.startswith("/"):
                continue
            if msg.author.id == bot.user.id:
                channel_memory[channel_id].append({"role": "assistant", "content": msg.content})
            else:
                channel_memory[channel_id].append({"role": "user", "content": f"{msg.author.display_name}: {msg.content}"})

        await interaction.followup.send(f"✅ Absorbed the last {len(channel_memory[channel_id])} messages for this chat.", ephemeral=True)
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

    channel_memory[channel_id] = []
    if user_id in global_user_memory:
        global_user_memory[user_id] = []

    await interaction.followup.send("🧠 Channel memory and global user context wiped! Clean slate.", ephemeral=True)

bot.run(DISCORD_TOKEN)