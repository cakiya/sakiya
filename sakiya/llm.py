import asyncio
import json

import discord
from openai import AsyncOpenAI

from . import memory, prompts, rag
from .formatting import sanitize_response
from .settings import settings

llm_client = AsyncOpenAI(base_url=settings.llm_url, api_key="koboldcpp")

# koboldcpp is single threaded/one at a time, need lock to make sure each call to the bot gets a response
llm_lock = asyncio.Lock()


def describe_location(channel: discord.abc.Messageable, author: discord.User | discord.Member) -> str:
    if isinstance(channel, discord.DMChannel):
        return f"Direct Messages with @{author.display_name}"
    if hasattr(channel, "guild") and channel.guild:
        return f"Server: '{channel.guild.name}', Channel: #{getattr(channel, 'name', 'chat')}"
    return "Private Group Chat"


def build_user_context(author: discord.User | discord.Member) -> str:
    memories = memory.global_user_memory.get(author.id)
    if not memories:
        return ""
    context = f"\n\n[Your recent cross-channel memories with @{author.display_name}]:\n"
    for mem in memories:
        context += f"- {mem}\n"
    return context


def build_system_prompt(location_desc: str, user_context: str, context: str, user_input: str) -> str:
    meta_system = f"{prompts.get_system_prompt()}\n\n[Current Chat Location: {location_desc}]{user_context}"
    if context:
        meta_system += f"\n\n[Background Knowledge Retrieved from Memory]:\n{context}"
    return meta_system + prompts.language_directive(user_input)


async def generate_bot_reply(
    channel: discord.abc.Messageable, author: discord.User | discord.Member, user_input: str
) -> str:
    channel_id = channel.id
    user_id = author.id

    # 1. RAG lookup
    context = await asyncio.to_thread(rag.get_relevant_context, user_input, settings.rag_top_k)

    # 2. Build human-readable channel/server context
    location_desc = describe_location(channel, author)

    # 3. Inject cross-channel global user memory
    user_context = build_user_context(author)

    # 4. Assemble the System Prompt
    meta_system = build_system_prompt(location_desc, user_context, context, user_input)

    formatted_user_msg = f"{prompts.display_name(author)}: {user_input}"

    # makes sure each call gets a response
    async with llm_lock:
        memory.append_channel_message(channel_id, {"role": "user", "content": formatted_user_msg})

        messages_payload = [{"role": "system", "content": meta_system}] + memory.channel_memory[channel_id]

        # 5. Call LLM
        print("\n=== INCOMING LLM PAYLOAD ===")
        print(json.dumps(messages_payload, indent=2, ensure_ascii=False))
        print("============================\n")

        response = await llm_client.chat.completions.create(
            model=settings.llm_model,
            messages=messages_payload,
            temperature=settings.temperature,
            frequency_penalty=settings.frequency_penalty,
            presence_penalty=settings.presence_penalty,
            extra_body={
                "min_p": settings.min_p,
                "top_p": settings.top_p,
                "rep_pen": settings.rep_pen,
                "dry_multiplier": settings.dry.multiplier,
                "dry_base": settings.dry.base,
                "dry_allowed_length": settings.dry.allowed_length,
                "dry_penalty_last_n": settings.dry.penalty_last_n
            }
        )
        print("\n\n=== RETURNING LLM RESPONSE ===")
        print(response)

        raw_reply = response.choices[0].message.content
        clean_reply = sanitize_response(raw_reply)

        memory.append_channel_message(channel_id, {"role": "assistant", "content": clean_reply})

    # 6. Update Global User Memory
    memory.append_user_exchange(
        user_id,
        f"They said: '{user_input}' | You replied: '{clean_reply}'"
    )

    return clean_reply
