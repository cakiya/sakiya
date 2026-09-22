import os
import asyncio
import numpy as np
import discord
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

# 1. Configuration
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
LLM_URL = "http://localhost:5001/v1"
PROMPT_PATH = ".prompt.txt"

# Store conversation history: { user_id : [list of message dicts] }
user_memory = {}
MAX_MEMORY = 10 # Only remember the last 10 messages to save VRAM

# 2. Local RAG Initialization
print("Loading embedding model and history... this may take a moment.")
embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

# Read and embed the chat logs once on startup, and load prompt from prompt.txt
try:
    with open(".personas/cakiya.txt", "r", encoding="utf-8") as f:
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

def get_relevant_context(query: str, top_k: int = 5) -> str:
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

# 4. Configure Discord Intents
intents = discord.Intents.default()
intents.message_content = True 
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if client.user in message.mentions:
        user_input = message.content.replace(f'<@{client.user.id}>', '').strip()
        
        async with message.channel.typing():
            try:
                # Execute the heavy math in a thread to keep the Discord heartbeat alive
                context = await asyncio.to_thread(get_relevant_context, user_input, 5)
                
                # Build final prompt with context
                final_prompt = system_prompt
                if context:
                    final_prompt += f"\n\n[Examples of your past messages to copy the style of]:\n{context}"

                # --- MEMORY MANAGEMENT ---
                # Initialize memory for this user if it doesn't exist
                if message.author.id not in user_memory:
                    user_memory[message.author.id] = []

                # Append the user's new message to their memory
                user_memory[message.author.id].append({"role": "user", "content": user_input})

                # Keep memory from getting too long
                if len(user_memory[message.author.id]) > MAX_MEMORY:
                    user_memory[message.author.id].pop(0)

                # Build the full payload: System Prompt + Chat History
                messages_payload = [{"role": "system", "content": final_prompt}] + user_memory[message.author.id]

                # Send the prompt to Koboldcpp asynchronously
                response = await llm_client.chat.completions.create(
                    model="local-model",
                    messages=messages_payload,
                    temperature=0.8
                )
                
                reply_text = response.choices[0].message.content
                
                # Append the bot's reply to the memory so it remembers its own answers
                user_memory[message.author.id].append({"role": "assistant", "content": reply_text})
                # -------------------------
                
                await message.channel.send(reply_text)
                
            except Exception as e:
                print("ERROR: ", e)
                await message.channel.send("Error communicating with local model.")

# Run the Discord bot
client.run(DISCORD_TOKEN)