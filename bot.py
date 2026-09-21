import os
import discord
from dotenv import load_dotenv
from openai import OpenAI

# 1. Configuration
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
LLM_URL = "http://localhost:5001/v1" # koboldcpp

# 2. Connect to local LM Studio server
llm_client = OpenAI(base_url=LLM_URL, api_key="totally_a_key")

# 3. Configure Discord Intents
intents = discord.Intents.default()
intents.message_content = True  # Required to read user messages
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):
    # Ignore messages sent by the bot itself to prevent infinite loops
    if message.author == client.user:
        return

    # Trigger only if the bot is mentioned (e.g., @MyBot hello!)
    if client.user in message.mentions:
        # Clean the message by removing the bot mention tag
        user_input = message.content.replace(f'<@{client.user.id}>', '').strip()
        
        # Show "Bot is typing..." in Discord while the GGUF model generates text
        async with message.channel.typing():
            try:
                # Send the prompt to your local GGUF model
                response = llm_client.chat.completions.create(
                    model="local-model",
                    messages=[
                        {"role": "system", "content": "You are a helpful and creative AI assistant in a Discord server."},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.8
                )
                
                reply_text = response.choices[0].message.content
                await message.channel.send(reply_text)
                
            except Exception as e:
                await message.channel.send(f"Error communicating with local model: {e}")

# Run the Discord bot
client.run(DISCORD_TOKEN)