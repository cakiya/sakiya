import os
import asyncio
import numpy as np
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
import koboldcpp  # Added import for your custom server module

# 1. Configuration
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
LLM_URL = "http://localhost:5001/v1"
PROMPT_PATH = os.path.join(BASE_DIR, "input/prompt.txt")
PERSONA_PATH = os.path.join(BASE_DIR, "input/personas/"+os.getenv("PERSONA", "default")+".txt")
KOBOLD_EXE_PATH = os.path.join(BASE_DIR, "koboldcpp.exe")
MODEL_PATH = os.path.join(BASE_DIR, "input/models/"+os.getenv("MODEL_NAME", "default")+".gguf")

# Start the background server
server_process = koboldcpp.start_server(KOBOLD_EXE_PATH, MODEL_PATH)

# Store conversation history: { user_id : [list of message dicts] }
user_memory = {}
MAX_MEMORY = 25 # Only remember the last 50 messages to save VRAM

# 2. Local RAG Initialization
print("Loading embedding model and history... this may take a moment.")
embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

# Read and embed the chat logs once on startup, and load prompt from prompt.txt
try:
    with open(PERSONA_PATH, "r", encoding="utf-8") as f:
        history_lines = [line.strip() for line in f if line.strip()]
    history_embeddings = embedder.encode(history_lines, convert_to_numpy=True)
    print(f"Successfully embedded {len(history_lines)} lines of context.")
except FileNotFoundError:
    print("WARNING: messages.txt not found. The bot will run without RAG context.")
    history_lines = []
    history_embeddings = np.array([])

try:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    print(f"Successfully loaded system prompt from {PROMPT_PATH}.")
except FileNotFoundError:
    print(f"WARNING: {PROMPT_PATH} not found. Using default system prompt.")
    system_prompt = "You are a helpful and creative AI assistant in a Discord server."

def get_relevant_context(query: str, top_k: int = 0) -> str: # rag turned off to 0 random entries
    """Synchronous function to perform the math calculation for vector search."""
    if not history_lines:
        return ""
    
    query_vec = embedder.encode([query], convert_to_numpy=True)
    scores = np.dot(history_embeddings, query_vec.T).squeeze()
    
    actual_k = min(top_k, len(scores))
    if actual_k == 0:
        return ""
        
    top_indices = np.argsort(scores)[::-1][:actual_k]
    matched = [history_lines[i] for i in top_indices]
    return "\n".join(matched)

# 3. Connect to local backend using the Async client
llm_client = AsyncOpenAI(base_url=LLM_URL, api_key="koboldcpp")

# 4. Configure Discord Intents & Switch to commands.Bot
intents = discord.Intents.default()
intents.message_content = True 
bot = commands.Bot(command_prefix="!", intents=intents)

# Core LLM generation logic abstracted to handle both DMs and Slash Commands seamlessly
async def generate_bot_reply(user_id: int, user_input: str) -> str:
    # Execute the heavy math in a thread to keep the Discord heartbeat alive
    context = await asyncio.to_thread(get_relevant_context, user_input, 5)
    
    # Build final prompt with context
    final_prompt = system_prompt
    if context:
        final_prompt += f"\n\n[Examples of your past messages to copy the style of]:\n{context}"

    # --- MEMORY MANAGEMENT ---
    if user_id not in user_memory:
        user_memory[user_id] = []

    user_memory[user_id].append({"role": "user", "content": user_input})

    if len(user_memory[user_id]) > MAX_MEMORY:
        user_memory[user_id].pop(0)

    messages_payload = [{"role": "system", "content": final_prompt}] + user_memory[user_id]

    response = await llm_client.chat.completions.create(
        model="local-model",
        messages=messages_payload,
        temperature=0.4  # Lowered to 0.4 to keep personality from shifting randomly
    )
    
    reply_text = response.choices[0].message.content
    
    user_memory[user_id].append({"role": "assistant", "content": reply_text})
    # -------------------------
    
    return reply_text

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
        
        async with message.channel.typing():
            try:
                reply_text = await generate_bot_reply(message.author.id, user_input)
                
                # Split the message into chunks of 1950 characters and send consecutively
                chunk_size = 1950
                for i in range(0, len(reply_text), chunk_size):
                    await message.channel.send(reply_text[i:i+chunk_size])
                
            except Exception as e:
                print("ERROR: ", e)
                await message.channel.send("Error communicating with local model.")


# 5. Define the User-Installable Slash Command
@bot.tree.command(name="sakiya", description="Summon sakiya anywhere as an app")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
@app_commands.describe(message="What do you want to say?")
async def sakiya(interaction: discord.Interaction, message: str):
    # Defer immediately so Discord knows the bot is "thinking..." and avoids the 3-second timeout
    await interaction.response.defer(thinking=True)
    
    try:
        reply_text = await generate_bot_reply(interaction.user.id, message.strip())
        
        # Split into chunks to handle the 2000-character limit
        chunk_size = 1950
        chunks = [reply_text[i:i + chunk_size] for i in range(0, len(reply_text), chunk_size)]
        
        # The first chunk uses followup to resolve the "thinking..." state
        await interaction.followup.send(chunks[0])
        
        # Additional chunks are sent if the message is super long
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)
            
    except Exception as e:
        print("ERROR: ", e)
        await interaction.followup.send("Error communicating with local model.")

@bot.tree.command(name="sync_memory", description="Absorb the last 25 messages in this chat into memory")
@app_commands.allowed_installs(guilds=True, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def sync_memory(interaction: discord.Interaction):
    
    # 1. Permission Check (Optional: uncomment and add your ID if you want to lock it)
    # ALLOWED_USER_IDS = [YOUR_DISCORD_USER_ID]
    # if interaction.user.id not in ALLOWED_USER_IDS:
    #     await interaction.response.send_message("No permission.", ephemeral=True)
    #     return

    # Defer ephemerally so only you see the success/fail message
    await interaction.response.defer(ephemeral=True)

    try:
        # 2. Fetch the last 25 messages
        # Using list comprehension with an async for-loop is the modern discord.py standard
        messages = [message async for message in interaction.channel.history(limit=25)]
        
        # Messages load from newest to oldest; reverse them so the AI reads chronologically
        messages.reverse()

        user_id = interaction.user.id
        user_memory[user_id] = []

        # 3. Parse messages into the AI's memory format
        for msg in messages:
            # Skip completely empty messages (e.g., just an image) or slash command text
            if not msg.content or msg.content.startswith("/"):
                continue
                
            if msg.author.id == bot.user.id:
                user_memory[user_id].append({"role": "assistant", "content": msg.content})
            else:
                # Appending the user's display name helps the AI understand who said what in group chats
                user_memory[user_id].append({"role": "user", "content": f"{msg.author.display_name}: {msg.content}"})

        await interaction.followup.send(f"✅ Successfully absorbed the last {len(user_memory[user_id])} messages into memory!", ephemeral=True)

    except discord.Forbidden:
        await interaction.followup.send("❌ I do not have permission to read message history in this channel.", ephemeral=True)
    except Exception as e:
        print("ERROR: ", e)
        await interaction.followup.send("❌ Something went wrong reading the chat history.", ephemeral=True)

# Run the Discord bot
bot.run(DISCORD_TOKEN)